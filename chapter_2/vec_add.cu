#include <cuda_runtime.h>
#include <iostream>
#include <vector>

using namespace std;

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file, const int line)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Runtime Error at: %s: %d\n", file, line);
        fprintf(stderr, "%s %s\n", func, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__global__ void vec_add(float *A, float *B, float *C, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i]; 
    }
}

int main() {
    int n = 1000; 
    size_t bytes = n * sizeof(float);

    vector<float> h_A(n);
    vector<float> h_B(n);
    vector<float> h_C(n);

    for (int i = 0; i < n; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    float *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, bytes));

    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    vec_add<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
    
    // 检查核函数是否启动报错
    CHECK_CUDA_ERROR(cudaGetLastError());
    // 等待 GPU 做完
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    printf("Result verification:\n");
    for (int i = 0; i < 5; i++) {
        printf("%.1f + %.1f = %.1f\n", h_A[i], h_B[i], h_C[i]);
    }

    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));

    return 0;
}