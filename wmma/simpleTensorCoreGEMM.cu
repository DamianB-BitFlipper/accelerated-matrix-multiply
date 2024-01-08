#include <iostream>
#include <cstdint>

#include <curand.h>
#include <cublas_v2.h>
#include <mma.h>

using namespace nvcuda;

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
    int32_t warpI = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int32_t warpJ = blockIdx.y * blockDim.y + threadIdx.y;

    // Define the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

    wmma::fill_fragment(acc_frag, 0.0f);
}

int32_t main() {
    float* a_fp32;
    float* b_fp32;
    half* a_fp16;
    half* b_fp16;
    half* c_wmma;

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
    cudaMalloc(&a_fp16, MATRIX_M * MATRIX_K * sizeof(half));
    cudaMalloc(&b_fp16, MATRIX_K * MATRIX_N * sizeof(half));

    // Curand does not support `half`, so generate random `float` and then convert to `half`
    {
        curandGenerateUniform(randGen, a_fp32, MATRIX_M * MATRIX_K);
        curandGenerateUniform(randGen, b_fp32, MATRIX_K * MATRIX_N);

        dim3 gridDim;
        dim3 blockDim;
    }

    // Clean up
    curandDestroyGenerator(randGen);
    cudaEventDestroy(startWMMA);
    cudaEventDestroy(stopWMMA);

    return 0;
}
