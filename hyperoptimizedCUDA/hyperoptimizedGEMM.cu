// Code adapted from: https://github.com/cwpearson/nvidia-performance-tools

#include <iostream>
#include <iomanip>
#include <cstdint>
#include <memory>
#include <algorithm>
#include <random>
#include <chrono>

#include <curand.h>
#include <cuda_fp16.h>
//#include <nvToolsExt.h>

// Local includes
#include <argparse/argparse.hpp>
#include <common.hpp>

/* NOTE: All matrices are in column major order
 */
__global__ void hyperoptimizedMatrixMultiply(
    const half *a,
    const half *b,
    const half *bias,
    half *__restrict__ c_out,
    const int32_t M,
    const int32_t N,
    const int32_t K,
    float alpha,
    float beta) {
#define A(_i, _j) a[(_i) + (_j)*M]
#define B(_i, _j) b[(_i) + (_j)*K]
#define BIAS(_i, _j) bias[(_i) + (_j)*M]
#define C(_i, _j) c_out[(_i) + (_j)*M]

    // Convert the `alpha` and `beta` weights into half precision
    const half alpha_fp16{ __float2half(alpha) };
    const half beta_fp16{ __float2half(beta) };

    const int32_t col{ static_cast<int32_t>(blockDim.x * blockIdx.x + threadIdx.x) };
    const int32_t row{ static_cast<int32_t>(blockDim.y * blockIdx.y + threadIdx.y) };

    for (int32_t i{ row }; i < M; i += gridDim.y * blockDim.y) {
        for (int32_t j{ col }; j < N; j += gridDim.x * blockDim.x) {
            float acc{ 0.0f };
            for (int32_t k{ 0 }; k < K; ++k) {
                acc += static_cast<float>(A(i, k) * B(k, j));
            }
            C(i, j) = alpha_fp16 * __float2half(acc) + beta_fp16 * BIAS(i, j);
        }
    }

#undef A
#undef B
#undef BIAS
#undef C
}

int main(int argc, char **argv) {
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

    // Allocate the device memory for the matrices
    cudaMalloc(&a_fp32, matrixM * matrixK * sizeof(float));
    cudaMalloc(&b_fp32, matrixK * matrixN * sizeof(float));
    cudaMalloc(&bias_fp32, matrixM * matrixN * sizeof(float));
    cudaMalloc(&c_fp32, matrixM * matrixN * sizeof(float));
    cudaMalloc(&a_fp16, matrixM * matrixK * sizeof(half));
    cudaMalloc(&b_fp16, matrixK * matrixN * sizeof(half));
    cudaMalloc(&bias_fp16, matrixM * matrixN * sizeof(half));
    cudaMalloc(&c_fp16, matrixM * matrixN * sizeof(half));

    float* a_host;
    float* b_host;
    float* bias_host;
    float* c_host;
    float* c_cuda_host;
    cudaHostAlloc(&a_host, matrixM * matrixK * sizeof(float), 0);
    cudaHostAlloc(&b_host, matrixK * matrixN * sizeof(float), 0);
    cudaHostAlloc(&bias_host, matrixM * matrixN * sizeof(float), 0);
    cudaHostAlloc(&c_host, matrixM * matrixN * sizeof(float), 0);
    cudaHostAlloc(&c_cuda_host, matrixM * matrixN * sizeof(float), 0);

    // Use std::generate to fill the host input arrays with random float values
    std::mt19937 randGen{ std::random_device{}() };
    std::uniform_real_distribution<float> dis(-420.0f, 420.0f);
    std::generate(a_host, a_host + matrixM * matrixK, [&]() { return dis(randGen); });
    std::generate(b_host, b_host + matrixK * matrixN, [&]() { return dis(randGen); });
    std::generate(bias_host, bias_host + matrixN * matrixM, [&]() { return dis(randGen); });

    // Clear the contents of `c` matrices
    cudaMemset(c_fp16, __float2half(0.0f), matrixM * matrixN * sizeof(half));
    std::fill(c_host, c_host + matrixM * matrixN, 0.0f);

    // Create and initialize the CUDA events
    cudaEvent_t startCUDA, stopCUDA;
    cudaEventCreate(&startCUDA);
    cudaEventCreate(&stopCUDA);

    std::cout << "Transfer to GPU" << std::endl;

    // Copy the random constents of the host matrices to the device float matrices
    //nvtxRangePush("host-to-device");
    cudaMemcpy(
        a_fp32, a_host, matrixM * matrixK * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(
        b_fp32, b_host, matrixK * matrixN * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(
        bias_fp32, bias_host, matrixM * matrixN * sizeof(float), cudaMemcpyHostToDevice);
    //nvtxRangePop();

    std::cout << "Converting to half" << std::endl;

    // Convert the device float matrices to half matrices
    //nvtxRangePush("float-to-half");
    dim3 blockDimConvert{ 256, 1, 1 };
    dim3 gridDimConvert{
        (static_cast<uint32_t>(matrixM * matrixN + blockDimConvert.x - 1) / blockDimConvert.x),
            1, 1 };
    convertFp32ToFp16<<<gridDimConvert, blockDimConvert>>>(a_fp32, a_fp16, matrixM * matrixK);
    convertFp32ToFp16<<<gridDimConvert, blockDimConvert>>>(b_fp32, b_fp16, matrixK * matrixN);
    convertFp32ToFp16<<<gridDimConvert, blockDimConvert>>>(bias_fp32, bias_fp16, matrixM * matrixN);
    //nvtxRangePop();

    // Some useful prints
    std::cout << "M = " << matrixM << ", "
              << "N = " << matrixN << ", "
              << "K = " << matrixK << ", "
              << std::fixed << std::setprecision(2)
              << "alpha = " << alpha << ", "
              << "beta = " << beta << std::endl;

    std::cout << "Running with CUDA" << std::endl;

    float cudaElapsedTime{ 0.0f };

    // GPU kernel launch parameters
    dim3 dimBlockCompute{ 32, 32 };
    dim3 dimGridCompute{
        (matrixN + dimBlockCompute.x - 1) / dimBlockCompute.x,
            (matrixM + dimBlockCompute.y - 1) / dimBlockCompute.y };

    // Perform the kernel matrix multiplication `nWarmup + nIters` times
    // Check for correctness on the first time.
    // Record the time after nWarmup runs complete.
    for (int i = 0; i < nIters + nWarmup; ++i) {
        // Launch the kernel matrix multiplication kernel
        //nvtxRangePush("kernel");
        cudaEventRecord(startCUDA);
        hyperoptimizedMatrixMultiply<<<dimGridCompute, dimBlockCompute>>>(
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
        //nvtxRangePop();

        // Launch the kernel to convert the `c_fp16`
        convertFp16ToFp32<<<gridDimConvert, blockDimConvert>>>(c_fp16, c_fp32, matrixM * matrixN);

        // Copy the result once for checking
        if (check && i == 0) {
            // Copy the result to the `c_cuda_host`
            cudaMemcpy(
                c_cuda_host, c_fp32, matrixM * matrixN * sizeof(float), cudaMemcpyDeviceToHost);
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
            a_host,
            b_host,
            bias_host,
            c_host,
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
    cudaFreeHost(a_host);
    cudaFreeHost(b_host);
    cudaFreeHost(bias_host);
    cudaFreeHost(c_host);
    cudaFreeHost(c_cuda_host);

    return 0;
}
