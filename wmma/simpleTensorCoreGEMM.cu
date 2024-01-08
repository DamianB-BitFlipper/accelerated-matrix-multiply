#include <iostream>
#include <iomanip>
#include <cstdint>

#include <curand.h>
#include <cublas_v2.h>
#include <mma.h>

using namespace nvcuda;

const int32_t WARP_SIZE{ 32 };

// Must be multiples of 16 for wmma code to work
const int32_t MATRIX_M{ 16384 };
const int32_t MATRIX_N{ 16384 };
const int32_t MATRIX_K{ 16384 };

// The only dimensions currently supported by WMMA
const int32_t WMMA_M{ 16 };
const int32_t WMMA_N{ 16 };
const int32_t WMMA_K{ 16 };

__global__ void wmma_matmul(
    half* a,
    half* b,
    half* bias,
    half* c_out,
    int32_t M,
    int32_t N,
    int32_t K,
    float alpha,
    float beta) {
    // Convert the `alpha` and `beta` weights into half precision
    const half alpha_fp16{ __float2half(alpha) };
    const half beta_fp16{ __float2half(beta) };

    // Tile using a 2D grid
    int32_t warpI = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int32_t warpJ = blockIdx.y * blockDim.y + threadIdx.y;

    // Define the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> bias_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    const int32_t aRow{ WMMA_M * warpJ };
    const int32_t bCol{ WMMA_N * warpI };

    for (int32_t i{ 0 }; i < K; i += WMMA_K) {
        const int32_t aCol{ i };
        const int32_t bRow{ i };

        // Bounds checking
        if (aRow < M && bCol < N) {
            // Fill the fragments noting the column-major memory format
            wmma::load_matrix_sync(a_frag, a + aCol * M + aRow, M);
            wmma::load_matrix_sync(b_frag, b + bCol * K + bRow, K);

            // Perform the matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Load and add in the bias
    wmma::load_matrix_sync(bias_frag, bias + bCol * M + aRow, M, wmma::mem_col_major);

#pragma unroll
    for (int32_t i{ 0 }; i < acc_frag.num_elements; i++) {
        c_frag.x[i] = alpha_fp16 * acc_frag.x[i] + beta_fp16 * bias_frag.x[i];
    }

    // Store the resulting output
    wmma::store_matrix_sync(c_out + bCol * M + aRow, c_frag, M, wmma::mem_col_major);
}

__global__ void convertFp32ToFp16(float* in, half* out, const int32_t len) {
    int32_t idx{ static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x) };
    if (idx < len) {
        out[idx] = static_cast<half>(in[idx]);
    }
}

int32_t main() {
    float* a_fp32;
    float* b_fp32;
    float* bias_fp32;

    half* a_fp16;
    half* b_fp16;
    half* bias_fp16;
    half* c_fp16;

    const float alpha{ 2.0f };
    const float beta{ 2.0f };

    // Create and initialize the CUDA random number generator
    curandGenerator_t randGen;
    curandCreateGenerator(&randGen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(randGen, 69);

    // Create and initialize the CUDA events
    cudaEvent_t startWMMA, stopWMMA;
    cudaEventCreate(&startWMMA);
    cudaEventCreate(&stopWMMA);

    // Allocate the memory for the matrices
    cudaMalloc(&a_fp32, MATRIX_M * MATRIX_K * sizeof(float));
    cudaMalloc(&b_fp32, MATRIX_K * MATRIX_N * sizeof(float));
    cudaMalloc(&bias_fp32, MATRIX_M * MATRIX_N * sizeof(float));
    cudaMalloc(&a_fp16, MATRIX_M * MATRIX_K * sizeof(half));
    cudaMalloc(&b_fp16, MATRIX_K * MATRIX_N * sizeof(half));
    cudaMalloc(&bias_fp16, MATRIX_M * MATRIX_N * sizeof(half));
    cudaMalloc(&c_fp16, MATRIX_M * MATRIX_N * sizeof(half));

    // Clear the contents of `c_fp16`
    cudaMemset(c_fp16, __float2half(0.0f), MATRIX_M * MATRIX_N * sizeof(half));

    // Curand does not support `half`, so generate random `float` and then convert to `half`
    {
        curandGenerateUniform(randGen, a_fp32, MATRIX_M * MATRIX_K);
        curandGenerateUniform(randGen, b_fp32, MATRIX_K * MATRIX_N);
        curandGenerateUniform(randGen, bias_fp32, MATRIX_K * MATRIX_N);

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

    // Some useful prints
    std::cout << "M = " << MATRIX_M << ", ";
    std::cout << "N = " << MATRIX_N << ", ";
    std::cout << "K = " << MATRIX_K << ", ";
    std::cout << std::fixed << std::setprecision(2) << "alpha = " << alpha << ", ";
    std::cout << std::fixed << std::setprecision(2) << "beta = " << beta << std::endl;

    std::cout << "Running with WMMA" << std::endl;

    // Perform the WMMA matrix multiplication
    {
        dim3 blockDim{ 256, 4, 1 };
        dim3 gridDim{
            (MATRIX_N + (WMMA_N * blockDim.x / WARP_SIZE) - 1) / (WMMA_N * blockDim.x / WARP_SIZE),
            (MATRIX_M + (WMMA_M * blockDim.y) - 1) / (WMMA_M * blockDim.y),
            1 };

        // Launch the WMMA matrix multiplication kernel
        cudaEventRecord(startWMMA);
        wmma_matmul<<<gridDim, blockDim>>>(
            a_fp16,
            b_fp16,
            bias_fp16,
            c_fp16,
            MATRIX_M,
            MATRIX_N,
            MATRIX_K,
            alpha,
            beta);
        cudaEventRecord(stopWMMA);
        cudaEventSynchronize(stopWMMA);
    }

    // Clean up
    curandDestroyGenerator(randGen);
    cudaEventDestroy(startWMMA);
    cudaEventDestroy(stopWMMA);

    // Free all of the allocated memory
    cudaFree(a_fp32);
    cudaFree(b_fp32);
    cudaFree(bias_fp32);
    cudaFree(a_fp16);
    cudaFree(b_fp16);
    cudaFree(bias_fp16);
    cudaFree(c_fp16);

    return 0;
}
