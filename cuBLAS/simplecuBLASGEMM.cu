#include <iostream>
#include <iomanip>
#include <cstdint>
#include <memory>
#include <algorithm>
#include <chrono>

#include <curand.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

// Local includes
#include <argparse/argparse.hpp>
#include <common.hpp>

// Dimensions currently supported by cuBLAS
const int32_t CUBLAS_M{ 16 };
const int32_t CUBLAS_N{ 16 };
const int32_t CUBLAS_K{ 16 };

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

    // The matrix dimensions must be multiples of CUBLAS_M, CUBLAS_N, CUBLAS_K respectively to work
    if (matrixM % CUBLAS_M != 0 ||
        matrixN % CUBLAS_N != 0 ||
        matrixK % CUBLAS_K != 0) {
        std::cerr << "The matrix dimensions must be multiples of: ("
                  << CUBLAS_M << ", " << CUBLAS_N << ", " << CUBLAS_K << ")"
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

    std::unique_ptr<float[]> a_host{ new float[matrixM * matrixK] };
    std::unique_ptr<float[]> b_host{ new float[matrixK * matrixN] };
    std::unique_ptr<float[]> bias_host{ new float[matrixM * matrixN] };
    std::unique_ptr<float[]> c_host{ new float[matrixM * matrixN] };
    std::unique_ptr<float[]> c_cuda_host{ new float[matrixM * matrixN] };

    // Clear the contents of `c` matrices
    cudaMemset(c_fp16, __float2half(0.0f), matrixM * matrixN * sizeof(half));
    std::fill(c_host.get(), c_host.get() + matrixM * matrixN, 0.0f);

    // Initialize the cuBLAS handle to use tensor cores
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);
    cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH);

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

    std::cout << "Running with cuBLAS" << std::endl;

    float cudaElapsedTime{ 0.0f };

    // The kernel parameters
    dim3 blockDimConvert{ 256, 1, 1 };
    dim3 gridDimConvert{
        (matrixM * matrixN + blockDimConvert.x - 1) / blockDimConvert.x, 1, 1 };

    // Perform the cuBLAS matrix multiplication `nWarmup + nIters` times
    // Check for correctness on the first time.
    // Record the time after nWarmup runs complete.
    for (int32_t i{ 0 }; i < nWarmup + nIters; i++) {
        // Launch the cuBLAS matrix multiplication kernel
        cudaEventRecord(startCUDA);
        cublasGemmEx(
            cublasHandle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            matrixM,
            matrixN,
            matrixK,
            &alpha,
            a_fp16,
            CUDA_R_16F,
            matrixM,
            b_fp16,
            CUDA_R_16F,
            matrixK,
            &beta,
            c_fp16,
            CUDA_R_16F,
            matrixM,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);

        // Add the bias in to the result. Treat the `bias_fp16` and `c_fp16`
        // as long vectors when adding them together
        cublasAxpyEx(
            cublasHandle,
            matrixM * matrixN,
            &beta,
            CUDA_R_32F,
            bias_fp16,
            CUDA_R_16F,
            1,
            c_fp16,
            CUDA_R_16F,
            1,
            CUDA_R_32F);
        cudaEventRecord(stopCUDA);
        cudaEventSynchronize(stopCUDA);

        // Convert the `c_fp16` to `c_fp32`
        dim3 blockDimConvert{ 256, 1, 1 };
        dim3 gridDimConvert{
            (matrixM * matrixN + blockDimConvert.x - 1) / blockDimConvert.x, 1, 1 };

        // Launch the kernel to convert the `c_fp16`
        convertFp16ToFp32<<<gridDimConvert, blockDimConvert>>>(c_fp16, c_fp32, matrixM * matrixN);

        // Reset the contents of `c_fp16` matrices
        cudaMemset(c_fp16, __float2half(0.0f), matrixM * matrixN * sizeof(half));

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
    cublasDestroy(cublasHandle);
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
