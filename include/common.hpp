#pragma once

#include <cstdint>
#include <cmath>

#include <curand.h>
#include <cuda_fp16.h>

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

void fillMatricesRand(
    curandGenerator_t randGen,
    float* a_fp32,
    float* b_fp32,
    float* bias_fp32,
    half* a_fp16,
    half* b_fp16,
    half* bias_fp16,
    std::unique_ptr<float[]>& a_host,
    std::unique_ptr<float[]>& b_host,
    std::unique_ptr<float[]>& bias_host,
    const int32_t M,
    const int32_t N,
    const int32_t K) {
    // Curand does not support `half`, so generate random `float` and then convert to `half`
    curandGenerateUniform(randGen, a_fp32, M * K);
    curandGenerateUniform(randGen, b_fp32, K * N);
    curandGenerateUniform(randGen, bias_fp32, M * N);

    const int32_t BLOCK_SIZE{ 256 };
    dim3 gridDimA{ static_cast<uint32_t>((M * K + BLOCK_SIZE - 1) / BLOCK_SIZE), 1, 1 };
    dim3 gridDimB{ static_cast<uint32_t>((K * N + BLOCK_SIZE - 1) / BLOCK_SIZE), 1, 1 };
    dim3 gridDimBias{ static_cast<uint32_t>((M * N + BLOCK_SIZE - 1) / BLOCK_SIZE), 1, 1 };
    dim3 blockDim{ BLOCK_SIZE, 1, 1 };

    // Create and initialize the CUDA streams
    cudaStream_t stream1, stream2, stream3;

    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);

    // Launch both kernels to convert the `a_fp32` and `b_fp32`
    convertFp32ToFp16<<<gridDimA, blockDim, 0, stream1>>>(a_fp32, a_fp16, M * K);
    convertFp32ToFp16<<<gridDimB, blockDim, 0, stream2>>>(b_fp32, b_fp16, K * N);
    convertFp32ToFp16<<<gridDimBias, blockDim, 0, stream3>>>(
        bias_fp32, bias_fp16, M * N);

    // Wait for both streams
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    cudaStreamSynchronize(stream3);

    // Clean up
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);

    // Copy the random contents of the device float matrices to the host matrices
    cudaMemcpy(
        a_host.get(), a_fp32, M * K * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(
        b_host.get(), b_fp32, K * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(
        bias_host.get(), bias_fp32, M * N * sizeof(float), cudaMemcpyDeviceToHost);
}

int32_t compareMatrices(
    float* c_host,
    float* c_cuda_host,
    const int32_t M,
    const int32_t N,
    const int32_t K
    ) {
    // Usa a 1% relative tolerance
    int32_t errors = 0;
    for (int32_t i = 0; i < M * N; i++) {
        float v1 = c_host[i];
        float v2 = c_cuda_host[i];
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

    return errors;
}

/**
 * A `unique_ptr` version of the function.
 */
int32_t compareMatrices(
    std::unique_ptr<float[]>& c_host,
    std::unique_ptr<float[]>& c_cuda_host,
    const int32_t M,
    const int32_t N,
    const int32_t K
    ) {
    return compareMatrices(c_host.get(), c_cuda_host.get(), M, N, K);
}
