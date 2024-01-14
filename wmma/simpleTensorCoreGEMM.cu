#include <iostream>
#include <iomanip>
#include <cstdint>
#include <memory>
#include <algorithm>
#include <chrono>

#include <curand.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>

// Local includes
#include <argparse/argparse.hpp>
#include <common.hpp>

using namespace nvcuda;

const int32_t WARP_SIZE{ 32 };

// The only dimensions currently supported by WMMA
const int32_t WMMA_M{ 16 };
const int32_t WMMA_N{ 16 };
const int32_t WMMA_K{ 16 };

__global__ void wmmaMatrixMultiply(
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

    // The matrix dimensions must be multiples of WMMA_M, WMMA_N, WMMA_K respectively to work
    if (matrixM % WMMA_M != 0 ||
        matrixN % WMMA_N != 0 ||
        matrixK % WMMA_K != 0) {
        std::cerr << "The matrix dimensions must be multiples of: ("
                  << WMMA_M << ", " << WMMA_N << ", " << WMMA_K << ")"
                  << std::endl;
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

    // Copy the random constents of the device float matrices to the host matrices
    cudaMemcpy(
        a_host.get(), a_fp32, matrixM * matrixK * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(
        b_host.get(), b_fp32, matrixK * matrixN * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(
        bias_host.get(), bias_fp32, matrixM * matrixN * sizeof(float), cudaMemcpyDeviceToHost);

    // Some useful prints
    std::cout << "M = " << matrixM << ", "
              << "N = " << matrixN << ", "
              << "K = " << matrixK << ", "
              << std::fixed << std::setprecision(2)
              << "alpha = " << alpha << ", "
              << "beta = " << beta << std::endl;

    std::cout << "Running with WMMA CUDA" << std::endl;

    float cudaElapsedTime{ 0.0f };

    // The kernel parameters
    dim3 blockDimCompute{ 256, 4, 1 };
    dim3 gridDimCompute{
        (matrixN + (WMMA_N * blockDimCompute.x / WARP_SIZE) - 1) /
            (WMMA_N * blockDimCompute.x / WARP_SIZE),
            (matrixM + (WMMA_M * blockDimCompute.y) - 1) / (WMMA_M * blockDimCompute.y),
            1 };

    dim3 blockDimConvert{ 256, 1, 1 };
    dim3 gridDimConvert{
        (matrixM * matrixN + blockDimConvert.x - 1) / blockDimConvert.x, 1, 1 };

    // Perform the WMMA matrix multiplication `nWarmup + nIters` times
    // Check for correctness on the first time.
    // Record the time after nWarmup runs complete.
    for (int32_t i{ 0 }; i < nWarmup + nIters; i++) {
        // Launch the WMMA matrix multiplication kernel
        cudaEventRecord(startCUDA);
        wmmaMatrixMultiply<<<gridDimCompute, blockDimCompute>>>(
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
