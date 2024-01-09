#include <iostream>
#include <iomanip>
#include <cstdint>
#include <memory>
#include <algorithm>
#include <chrono>

#include <curand.h>
#include <cuda_fp16.h>

const int32_t BLOCK_WIDTH{ 16 };

const float ALPHA{ 2.0f };
const float BETA{ 30.0f };

// Must be multiples of 16 for wmma code to work
const int32_t MATRIX_M{ 1024 };
const int32_t MATRIX_N{ 1024 };
const int32_t MATRIX_K{ 1024 };

__global__ void tiledMatrixMultiply(
    half* a,
    half* b,
    half* bias,
    half* c_out,
    int32_t M,
    int32_t N,
    int32_t K,
    float alpha,
    float beta) {
    // Dynamically allocate shared memory for subTileA and subTileB
    extern __shared__ half sharedMemory[];
    half* subTileA{ sharedMemory };
    half* subTileB{ sharedMemory + BLOCK_WIDTH * BLOCK_WIDTH };

    // Convert the `alpha` and `beta` weights into half precision
    const half alpha_fp16{ __float2half(alpha) };
    const half beta_fp16{ __float2half(beta) };

    // Compute the number of sub-tiles to cover the input matrices
    const int32_t numSubTiles{ (K + BLOCK_WIDTH - 1) / BLOCK_WIDTH };

    // Extract the `row` and `col` locations in the resulting `c_out` matrix
    const int32_t row{ static_cast<int32_t>(blockIdx.y * blockDim.y + threadIdx.y) };
    const int32_t col{ static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x) };

    // Extract the respective row and column of the sub-tiles
    const int32_t subTileARow{ static_cast<int32_t>(threadIdx.y) };
    const int32_t subTileACol{ static_cast<int32_t>(threadIdx.x) };
    const int32_t subTileBRow{ static_cast<int32_t>(threadIdx.y) };
    const int32_t subTileBCol{ static_cast<int32_t>(threadIdx.x) };

    float dot_prod{ 0.0f };

    for (int32_t subtileIdx{ 0 }; subtileIdx < numSubTiles; subtileIdx++) {
        // Load the sub-tiles from the input matrices into shared memory.
        // Note: the data is stored in column-major order in `a` and `b`
        if (row < M && (subtileIdx * BLOCK_WIDTH + subTileACol) < K) {
            subTileA[subTileARow * BLOCK_WIDTH + subTileACol] =
                a[(subtileIdx * BLOCK_WIDTH + subTileACol) * M + row];
        } else {
            subTileA[subTileARow * BLOCK_WIDTH + subTileACol] = 0;
        }

        if ((subtileIdx * BLOCK_WIDTH + subTileBRow) < K && col < N) {
            subTileB[subTileBRow * BLOCK_WIDTH + subTileBCol] =
                b[col * K + subtileIdx * BLOCK_WIDTH + subTileBRow];
        } else {
            subTileB[subTileBRow * BLOCK_WIDTH + subTileBCol] = 0;
        }

        // Wait for all threads to finish loading the sub-tiles
        __syncthreads();

        // Compute the partial sum of the resulting `c_out` matrix
        for (int32_t idx{ 0 }; idx < BLOCK_WIDTH; idx++) {
            dot_prod += static_cast<float>(
                subTileA[subTileARow * BLOCK_WIDTH + idx] *
                subTileB[idx * BLOCK_WIDTH + subTileBCol]);
        }

        // Wait for all threads to finish the computation before loading the new shared memory
        __syncthreads();
    }

    // Add the `dot_prod` and `bias` and scale accordingly.
    // Note: the data is stored in column-major order in `c_out`
    if (row < M && col < N) {
        c_out[col * M + row] = alpha_fp16 * __float2half(dot_prod) + beta_fp16 * bias[col * M + row];
    }
}

__global__ void convertFp32ToFp16(float* in, half* out, const int32_t len) {
    int32_t idx{ static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x) };
    if (idx < len) {
        out[idx] = static_cast<half>(in[idx]);
    }
}

__global__ void convertFp16ToFp32(half* in, float* out, const int32_t len) {
    int32_t idx{ static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x) };
    if (idx < len) {
        out[idx] = static_cast<float>(in[idx]);
    }
}

void referenceMatrixMultiply(
    float* a,
    float* b,
    float* bias,
    float* c_out,
    int32_t M,
    int32_t N,
    int32_t K,
    float alpha,
    float beta) {
    // Note: All input/output matrices are in column-major memory order
    for (int32_t aRow{ 0 }; aRow < M; aRow++) {
        for (int32_t bCol{ 0 }; bCol < N; bCol++) {
            // Compute the value at `aRow, bCol`
            for (int32_t k{ 0 }; k < K; k++) {
                c_out[aRow + bCol * M] += a[aRow + k * M] * b[k + bCol * K];
            }

            // Scale the `a * b` result by `alpha`
            c_out[aRow + bCol * M] *= alpha;

            // Add in the `bias` scaled by `beta`
            c_out[aRow + bCol * M] += beta * bias[aRow + bCol * M];
        }
    }
}

int32_t main() {
    float* a_fp32;
    float* b_fp32;
    float* bias_fp32;
    float* c_fp32;

    half* a_fp16;
    half* b_fp16;
    half* bias_fp16;
    half* c_fp16;

    // Allocate the memory for the matrices
    cudaMalloc(&a_fp32, MATRIX_M * MATRIX_K * sizeof(float));
    cudaMalloc(&b_fp32, MATRIX_K * MATRIX_N * sizeof(float));
    cudaMalloc(&bias_fp32, MATRIX_M * MATRIX_N * sizeof(float));
    cudaMalloc(&c_fp32, MATRIX_M * MATRIX_N * sizeof(float));
    cudaMalloc(&a_fp16, MATRIX_M * MATRIX_K * sizeof(half));
    cudaMalloc(&b_fp16, MATRIX_K * MATRIX_N * sizeof(half));
    cudaMalloc(&bias_fp16, MATRIX_M * MATRIX_N * sizeof(half));
    cudaMalloc(&c_fp16, MATRIX_M * MATRIX_N * sizeof(half));

    std::unique_ptr<float[]> a_host{ new float[MATRIX_M * MATRIX_K * sizeof(float)] };
    std::unique_ptr<float[]> b_host{ new float[MATRIX_K * MATRIX_N * sizeof(float)] };
    std::unique_ptr<float[]> bias_host{ new float[MATRIX_M * MATRIX_N * sizeof(float)] };
    std::unique_ptr<float[]> c_host{ new float[MATRIX_M * MATRIX_N * sizeof(float)] };
    std::unique_ptr<float[]> c_tiled_host{ new float[MATRIX_M * MATRIX_N * sizeof(float)] };

    // Clear the contents of `c` matrices
    cudaMemset(c_fp16, __float2half(0.0f), MATRIX_M * MATRIX_N * sizeof(half));
    std::fill(c_host.get(), c_host.get() + MATRIX_M * MATRIX_N, 0.0f);

    // Create and initialize the CUDA random number generator
    curandGenerator_t randGen;
    curandCreateGenerator(&randGen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(randGen, 69);

    // Create and initialize the CUDA events
    cudaEvent_t startTiled, stopTiled;
    cudaEventCreate(&startTiled);
    cudaEventCreate(&stopTiled);

    // Create the timing variables for the reference CPU matrix multiply
    std::chrono::high_resolution_clock::time_point referenceStartTime, referenceEndTime;

    // Curand does not support `half`, so generate random `float` and then convert to `half`
    {
        curandGenerateUniform(randGen, a_fp32, MATRIX_M * MATRIX_K);
        curandGenerateUniform(randGen, b_fp32, MATRIX_K * MATRIX_N);
        curandGenerateUniform(randGen, bias_fp32, MATRIX_M * MATRIX_N);

        const int32_t BLOCK_SIZE{ 256 };
        dim3 gridDimA{ (MATRIX_M * MATRIX_K + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1 };
        dim3 gridDimB{ (MATRIX_K * MATRIX_N + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1 };
        dim3 gridDimBias{ (MATRIX_M * MATRIX_N + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1 };
        dim3 blockDim{ BLOCK_SIZE, 1, 1 };

        // Create and initialize the CUDA streams
        cudaStream_t stream1, stream2, stream3;

        cudaStreamCreate(&stream1);
        cudaStreamCreate(&stream2);
        cudaStreamCreate(&stream3);

        // Launch both kernels to convert the `a_fp32` and `b_fp32`
        convertFp32ToFp16<<<gridDimA, blockDim, 0, stream1>>>(a_fp32, a_fp16, MATRIX_M * MATRIX_K);
        convertFp32ToFp16<<<gridDimB, blockDim, 0, stream2>>>(b_fp32, b_fp16, MATRIX_K * MATRIX_N);
        convertFp32ToFp16<<<gridDimBias, blockDim, 0, stream3>>>(
            bias_fp32, bias_fp16, MATRIX_M * MATRIX_N);

        // Wait for both streams
        cudaStreamSynchronize(stream1);
        cudaStreamSynchronize(stream2);
        cudaStreamSynchronize(stream3);

        // Clean up
        cudaStreamDestroy(stream1);
        cudaStreamDestroy(stream2);
        cudaStreamDestroy(stream3);
    }

    // Copy the random constents of the device float matrices to the host matrices
    cudaMemcpy(
        a_host.get(), a_fp32, MATRIX_M * MATRIX_K * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(
        b_host.get(), b_fp32, MATRIX_K * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(
        bias_host.get(), bias_fp32, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost);

    // Some useful prints
    std::cout << "M = " << MATRIX_M << ", "
              << "N = " << MATRIX_N << ", "
              << "K = " << MATRIX_K << ", "
              << std::fixed << std::setprecision(2)
              << "alpha = " << ALPHA << ", "
              << "beta = " << BETA << std::endl;

    std::cout << "Running with Tiled CUDA" << std::endl;

    // Perform the tiled matrix multiplication
    {
        dim3 gridDim{
            (MATRIX_N + BLOCK_WIDTH - 1) / BLOCK_WIDTH,
            (MATRIX_M + BLOCK_WIDTH - 1) / BLOCK_WIDTH,
            1};
        dim3 blockDim{ BLOCK_WIDTH, BLOCK_WIDTH, 1 };
        const int32_t sharedMemSize{ 2 * BLOCK_WIDTH * BLOCK_WIDTH * sizeof(half) };

        // Launch the tiled matrix multiplication kernel
        cudaEventRecord(startTiled);
        tiledMatrixMultiply<<<gridDim, blockDim, sharedMemSize>>>(
            a_fp16,
            b_fp16,
            bias_fp16,
            c_fp16,
            MATRIX_M,
            MATRIX_N,
            MATRIX_K,
            ALPHA,
            BETA);
        cudaEventRecord(stopTiled);
        cudaEventSynchronize(stopTiled);

        // Convert the `c_fp16` to `c_fp32`
        dim3 blockDimConvert{ 256, 1, 1 };
        dim3 gridDimConvert{
            (MATRIX_M * MATRIX_N + blockDimConvert.x - 1) / blockDimConvert.x, 1, 1 };

        // Launch the kernel to convert the `c_fp16`
        convertFp16ToFp32<<<gridDimConvert, blockDimConvert>>>(c_fp16, c_fp32, MATRIX_M * MATRIX_N);

        // Copy the result to the `c_tiled_host`
        cudaMemcpy(
            c_tiled_host.get(), c_fp32, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost);
    }

    std::cout << "Running on CPU" << std::endl;

    // Perform the CPU matrix multiplication
    {
        referenceStartTime = std::chrono::high_resolution_clock::now();
        referenceMatrixMultiply(
            a_host.get(),
            b_host.get(),
            bias_host.get(),
            c_host.get(),
            MATRIX_M,
            MATRIX_N,
            MATRIX_K,
            ALPHA,
            BETA);
        referenceEndTime = std::chrono::high_resolution_clock::now();
    }

    // Compare the matrix outputs
    {
        // Usa a 1% relative tolerance
        int32_t errors = 0;
        for (int32_t i = 0; i < MATRIX_M * MATRIX_N; i++) {
            float v1 = c_host[i];
            float v2 = c_tiled_host[i];
            float diff  = fabs(v1 - v2);
            float relative_err = diff / v2;
            float eps = 0.01;
            if (relative_err >= eps) {
                errors++;
                if (errors < 10) {
                    std::cout << v1 << " " << v2 << std::endl;
                }
            }
        }

        if (errors > 0) {
            std::cout << "Tiled CUDA does not agree with reference! " << errors
                      << " errors!" << std::endl;
        } else {
            std::cout << "Results verified: reference and Tiled CUDA agree." << std::endl;
            float tiledTime;
            cudaEventElapsedTime(&tiledTime, startTiled, stopTiled);

            std::chrono::milliseconds referenceDuration{
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    referenceEndTime - referenceStartTime) };
            std::cout << std::fixed << std::setprecision(4)
                      << "Tiled CUDA took " << tiledTime << "ms\n"
                      << "reference took " << referenceDuration.count() << "ms" << std::endl;
        }
    }

    // Clean up
    curandDestroyGenerator(randGen);
    cudaEventDestroy(startTiled);
    cudaEventDestroy(stopTiled);

    // Free all of the allocated memory
    cudaFree(a_fp32);
    cudaFree(b_fp32);
    cudaFree(bias_fp32);
    cudaFree(c_fp32);
    cudaFree(a_fp16);
    cudaFree(b_fp16);
    cudaFree(bias_fp16);
    cudaFree(c_fp16);

    return 0;
}
