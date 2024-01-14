#include <iostream>
#include <iomanip>
#include <cstdint>
#include <memory>
#include <algorithm>
#include <chrono>

#include <curand.h>
#include <cuda_fp16.h>

// Local includes
#include <argparse/argparse.hpp>
#include <common.hpp>

const int32_t BLOCK_WIDTH{ 16 };

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



int32_t main(int argc, char *argv[]) {
    argparse::Parser parser;
    // default matrix sizes:
    // A: 1024 x 1024
    // B: 1024 x 1024
    // C: 1024 x 1024
    int32_t matrixM{ 1024 };
    int32_t matrixN{ 1024 };
    int32_t matrixK{ 1024 };

    float alpha{ 2.0f };
    float beta{ 30.0f };
    int32_t nIters{ 40 };
    int32_t nWarmup{ 10 };
    bool check{ false };
    parser.add_positional(matrixM);
    parser.add_positional(matrixN);
    parser.add_positional(matrixK);
    parser.add_option(alpha, "--alpha");
    parser.add_option(beta, "--beta");
    parser.add_option(nIters, "--iters");
    parser.add_option(nWarmup, "--warmup");
    parser.add_flag(check, "--check");

    if (!parser.parse(argc, argv)) {
        parser.help();
        exit(EXIT_FAILURE);
    }

    // Times 2 because of the multiplication and addition
    const int64_t FLOP{ 2 *
            static_cast<int64_t>(matrixM) *
            static_cast<int64_t>(matrixN) *
            static_cast<int64_t>(matrixK) };

    float* a_fp32;
    float* b_fp32;
    float* bias_fp32;
    float* c_fp32;

    half* a_fp16;
    half* b_fp16;
    half* bias_fp16;
    half* c_fp16;

    // Allocate the memory for the matrices
    cudaMalloc(&a_fp32, matrixM * matrixK * sizeof(float));
    cudaMalloc(&b_fp32, matrixK * matrixN * sizeof(float));
    cudaMalloc(&bias_fp32, matrixM * matrixN * sizeof(float));
    cudaMalloc(&c_fp32, matrixM * matrixN * sizeof(float));
    cudaMalloc(&a_fp16, matrixM * matrixK * sizeof(half));
    cudaMalloc(&b_fp16, matrixK * matrixN * sizeof(half));
    cudaMalloc(&bias_fp16, matrixM * matrixN * sizeof(half));
    cudaMalloc(&c_fp16, matrixM * matrixN * sizeof(half));

    std::unique_ptr<float[]> a_host{ new float[matrixM * matrixK * sizeof(float)] };
    std::unique_ptr<float[]> b_host{ new float[matrixK * matrixN * sizeof(float)] };
    std::unique_ptr<float[]> bias_host{ new float[matrixM * matrixN * sizeof(float)] };
    std::unique_ptr<float[]> c_host{ new float[matrixM * matrixN * sizeof(float)] };
    std::unique_ptr<float[]> c_cuda_host{ new float[matrixM * matrixN * sizeof(float)] };

    // Clear the contents of `c` matrices
    cudaMemset(c_fp16, __float2half(0.0f), matrixM * matrixN * sizeof(half));
    std::fill(c_host.get(), c_host.get() + matrixM * matrixN, 0.0f);

    // Create and initialize the CUDA random number generator
    curandGenerator_t randGen;
    curandCreateGenerator(&randGen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(randGen, 69);

    // Create and initialize the CUDA events
    cudaEvent_t startCUDA, stopCUDA;
    cudaEventCreate(&startCUDA);
    cudaEventCreate(&stopCUDA);

    // Fill the matrices with random values
    fillMatricesRand(randGen,
                     a_fp32, b_fp32, bias_fp32,
                     a_fp16, b_fp16, bias_fp16,
                     a_host, b_host, bias_host,
                     matrixM, matrixN, matrixK);

    // Some useful prints
    std::cout << "M = " << matrixM << ", "
              << "N = " << matrixN << ", "
              << "K = " << matrixK << ", "
              << std::fixed << std::setprecision(2)
              << "alpha = " << alpha << ", "
              << "beta = " << beta << std::endl;

    std::cout << "Running with Tiled CUDA" << std::endl;

    float cudaElapsedTime{ 0.0f };

    // The kernel parameters
    dim3 gridDimCompute{
        static_cast<uint32_t>((matrixN + BLOCK_WIDTH - 1) / BLOCK_WIDTH),
            static_cast<uint32_t>((matrixM + BLOCK_WIDTH - 1) / BLOCK_WIDTH),
            1};
    dim3 blockDimCompute{ BLOCK_WIDTH, BLOCK_WIDTH, 1 };
    const int32_t sharedMemSize{ 2 * BLOCK_WIDTH * BLOCK_WIDTH * sizeof(half) };

    dim3 blockDimConvert{ 256, 1, 1 };
    dim3 gridDimConvert{
        (static_cast<uint32_t>(matrixM * matrixN + blockDimConvert.x - 1) / blockDimConvert.x),
            1, 1 };

    // Perform the tiled matrix multiplication `nWarmup + nIters` times
    // Check for correctness on the first time.
    // Record the time after nWarmup runs complete.
    for (int32_t i{ 0 }; i < nWarmup + nIters; i++) {
        // Launch the tiled matrix multiplication kernel
        cudaEventRecord(startCUDA);
        tiledMatrixMultiply<<<gridDimCompute, blockDimCompute, sharedMemSize>>>(
            a_fp16,
            b_fp16,
            bias_fp16,
            c_fp16,
            matrixM,
            matrixN,
            matrixK,
            alpha,
            beta);
        cudaEventRecord(stopCUDA);
        cudaEventSynchronize(stopCUDA);

        // Launch the kernel to convert the `c_fp16`
        convertFp16ToFp32<<<gridDimConvert, blockDimConvert>>>(c_fp16, c_fp32, matrixM * matrixN);

        // Copy the result once for checking
        if (check && i == 0) {
            // Copy the result to the `c_cuda_host`
            cudaMemcpy(
                c_cuda_host.get(), c_fp32, matrixM * matrixN * sizeof(float), cudaMemcpyDeviceToHost);
        }

        // Record the runtime after the warmup runs
        if (i >= nWarmup) {
            float elapsed;
            cudaEventElapsedTime(&elapsed, startCUDA, stopCUDA);

            cudaElapsedTime += elapsed;
        }
    }

    // Perform the CPU matrix multiplication if `check` is enabled
    if (check) {
        std::cout << "Running on CPU" << std::endl;
        auto referenceStartTime{ std::chrono::high_resolution_clock::now() };
        referenceMatrixMultiply(
            a_host.get(),
            b_host.get(),
            bias_host.get(),
            c_host.get(),
            matrixM,
            matrixN,
            matrixK,
            alpha,
            beta);
        auto referenceEndTime{ std::chrono::high_resolution_clock::now() };

        // Compare the matrix outputs
        int32_t errors{ compareMatrices(c_host, c_cuda_host, matrixM, matrixN, matrixK) };

        if (errors > 0) {
            std::cout << "Tiled CUDA does not agree with reference! " << errors
                      << " errors!" << std::endl;
        } else {
            std::cout << "Results verified: reference and Tiled CUDA agree." << std::endl;
        }

        // Print the reference runtime
        std::chrono::milliseconds referenceDuration{
            std::chrono::duration_cast<std::chrono::milliseconds>(
                referenceEndTime - referenceStartTime) };

        const double gFlops{ (FLOP / (referenceDuration.count() / 1000)) / 1e9 };
        std::cout << "Reference took " << referenceDuration.count() << "ms, "
                  << gFlops << " GFlops" << std::endl;
    }

    // Print the average CUDA runtime
    const double gFlops{ (FLOP / ((cudaElapsedTime / nIters) / 1000)) / 1e9 };
    std::cout << std::fixed << std::setprecision(4)
              << "CUDA took " << (cudaElapsedTime / nIters) << "ms, "
              << gFlops << " GFlops" << std::endl;

    // Clean up
    curandDestroyGenerator(randGen);
    cudaEventDestroy(startCUDA);
    cudaEventDestroy(stopCUDA);

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
