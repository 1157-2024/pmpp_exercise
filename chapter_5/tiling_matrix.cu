#include <cuda_runtime.h>
#include <iostream>
#include <vector>

using namespace std;

#define TILE_WIDTH 16

__global__ void matrixMulKernel(float *M, float *N, float *P, int width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float Pvalue = 0;
    
    // 这里的循环次数是 width/TILE_WIDTH，要求 width 必须是 16 的倍数
    for (int ph = 0; ph < width / TILE_WIDTH; ++ph) {
        Mds[ty][tx] = M[row * width + ph * TILE_WIDTH + tx];
        Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * width + col];

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }

        __syncthreads();
    }

    P[row * width + col] = Pvalue;
}

int main() {
    // 1. 改小一点方便测试
    int width = 2048; 
    
    int num_elements = width * width;
    size_t size_bytes = num_elements * sizeof(float);

    cout << "Matrix Size: " << width << " x " << width << endl;
    cout << "Memory Usage (Host): " << (size_bytes * 3) / (1024.0 * 1024.0) << " MB" << endl;

    vector<float> M_h(num_elements);
    vector<float> N_h(num_elements);
    vector<float> Out_h(num_elements);

    // 初始化数据
    for (int i = 0; i < num_elements; i++) {
        M_h[i] = rand() % 10; 
        N_h[i] = rand() % 10;
    }

    // 申请显存 (使用字节大小)
    float *M_d, *N_d, *Out_d;
    cudaMalloc((void **)&M_d, size_bytes);
    cudaMalloc((void **)&N_d, size_bytes);
    cudaMalloc((void **)&Out_d, size_bytes);

    cudaMemcpy(M_d, M_h.data(), size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h.data(), size_bytes, cudaMemcpyHostToDevice);

    dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (width + blockSize.y - 1) / blockSize.y
    );

    matrixMulKernel<<<gridSize, blockSize>>>(M_d, N_d, Out_d, width);

    // 检查 Kernel 是否执行出错
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cout << "CUDA Error: " << cudaGetErrorString(err) << endl;
    }
    
    // 同步等待结果
    cudaDeviceSynchronize();

    cudaMemcpy(Out_h.data(), Out_d, size_bytes, cudaMemcpyDeviceToHost);

    // 验证一下 P[0][0]
    cout << "GPU Result Out[0] = " << Out_h[0] << endl;
    
    float res = 0;
    for (int k = 0; k < width; k++) {
        res += M_h[0 * width + k] * N_h[k * width + 0];
    }
    cout << "CPU Check  Out[0] = " << res << endl;

    if (abs(Out_h[0] - res) < 1e-3) {
        cout << "Test PASSED!" << endl;
    } else {
        cout << "Test FAILED!" << endl;
    }

    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(Out_d);

    return 0;
}