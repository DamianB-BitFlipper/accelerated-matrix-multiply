#include <iostream>
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
    half* c,
    int32_t M,
    int32_t N,
    int32_t K,
    float alpha,
    float beta) {

    // Tile using a 2D grid
    int32_t warpI = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int32_t warpJ = blockIdx.y * blockDim.y + threadIdx.y;

    // Define the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

    wmma::fill_fragment(acc_frag, 0.0f);
}

__global__ void convertFp32ToFp16(float* in, half* out, const int32_t len) {
    int32_t idx{ blockIdx.x * blockDim.x + threadIdx.x };
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

    // Clean up
    curandDestroyGenerator(randGen);
    cudaEventDestroy(startWMMA);
    cudaEventDestroy(stopWMMA);

    cudaFree(a_fp32);
    cudaFree(b_fp32);
    cudaFree(bias_fp32);
    cudaFree(a_fp16);
    cudaFree(b_fp16);
    cudaFree(bias_fp16);
    cudaFree(c_fp16);

    return 0;
}
