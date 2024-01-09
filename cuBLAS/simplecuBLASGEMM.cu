#include <iostream>
#include <iomanip>
#include <cstdint>
#include <memory>
#include <algorithm>
#include <chrono>

#include <curand.h>
#include <cublas_v2.h>

const float ALPHA{ 2.0f };
const float BETA{ 30.0f };

// Must be multiples of 16 for wmma code to work
const int32_t MATRIX_M{ 1024 };
const int32_t MATRIX_N{ 1024 };
const int32_t MATRIX_K{ 1024 };

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
    std::unique_ptr<float[]> c_cublas_host{ new float[MATRIX_M * MATRIX_N * sizeof(float)] };

    // Clear the contents of `c` matrices
    cudaMemset(c_fp16, __float2half(0.0f), MATRIX_M * MATRIX_N * sizeof(half));
    std::fill(c_host.get(), c_host.get() + MATRIX_M * MATRIX_N, 0.0f);

    // Initialize the cuBLAS handle to use tensor cores
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);
    cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH);

    // Create and initialize the CUDA random number generator
    curandGenerator_t randGen;
    curandCreateGenerator(&randGen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(randGen, 69);

    // Create and initialize the CUDA events
    cudaEvent_t startCublas, stopCublas;
    cudaEventCreate(&startCublas);
    cudaEventCreate(&stopCublas);

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

    std::cout << "Running with cuBLAS" << std::endl;

    // Perform the cuBLAS matrix multiplication
    {
        // Warm up cuBLAS
        cublasGemmEx(
            cublasHandle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            MATRIX_M,
            MATRIX_N,
            MATRIX_K,
            &ALPHA,
            a_fp16,
            CUDA_R_16F,
            MATRIX_M,
            b_fp16,
            CUDA_R_16F,
            MATRIX_K,
            &ALPHA,
            c_fp16,
            CUDA_R_16F,
            MATRIX_M,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);

        cublasAxpyEx(
            cublasHandle,
            MATRIX_M * MATRIX_N,
            &BETA,
            CUDA_R_32F,
            bias_fp16,
            CUDA_R_16F,
            1,
            c_fp16,
            CUDA_R_16F,
            1,
            CUDA_R_32F);

        // Reset the contents of `c_fp16` matrices
        cudaMemset(c_fp16, __float2half(0.0f), MATRIX_M * MATRIX_N * sizeof(half));

        // Launch the cuBLAS matrix multiplication kernel
        cudaEventRecord(startCublas);
        cublasGemmEx(
            cublasHandle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            MATRIX_M,
            MATRIX_N,
            MATRIX_K,
            &ALPHA,
            a_fp16,
            CUDA_R_16F,
            MATRIX_M,
            b_fp16,
            CUDA_R_16F,
            MATRIX_K,
            &BETA,
            c_fp16,
            CUDA_R_16F,
            MATRIX_M,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);

        // Add the bias in to the result. Treat the `bias_fp16` and `c_fp16`
        // as long vectors when adding them together
        cublasAxpyEx(
            cublasHandle,
            MATRIX_M * MATRIX_N,
            &BETA,
            CUDA_R_32F,
            bias_fp16,
            CUDA_R_16F,
            1,
            c_fp16,
            CUDA_R_16F,
            1,
            CUDA_R_32F);
        cudaEventRecord(stopCublas);
        cudaEventSynchronize(stopCublas);

        // Convert the `c_fp16` to `c_fp32`
        dim3 blockDimConvert{ 256, 1, 1 };
        dim3 gridDimConvert{
            (MATRIX_M * MATRIX_N + blockDimConvert.x - 1) / blockDimConvert.x, 1, 1 };

        // Launch the kernel to convert the `c_fp16`
        convertFp16ToFp32<<<gridDimConvert, blockDimConvert>>>(c_fp16, c_fp32, MATRIX_M * MATRIX_N);

        // Copy the result to the `c_cublas_host`
        cudaMemcpy(
            c_cublas_host.get(), c_fp32, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost);
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
            float v2 = c_cublas_host[i];
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
            std::cout << "cuBLAS does not agree with reference! " << errors
                      << " errors!" << std::endl;
        } else {
            std::cout << "Results verified: reference and cuBLAS agree." << std::endl;
            float cublasTime;
            cudaEventElapsedTime(&cublasTime, startCublas, stopCublas);

            std::chrono::milliseconds referenceDuration{
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    referenceEndTime - referenceStartTime) };
            std::cout << std::fixed << std::setprecision(4)
                      << "cuBLAS took " << cublasTime << "ms\n"
                      << "reference took " << referenceDuration.count() << "ms" << std::endl;
        }
    }

    // Clean up
    curandDestroyGenerator(randGen);
    cublasDestroy(cublasHandle);
    cudaEventDestroy(startCublas);
    cudaEventDestroy(stopCublas);

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
