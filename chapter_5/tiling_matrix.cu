#include <cuda_runtime.h>
#include <iostream>
#include <vector>

using namespace std;
#define TILE_WIDTH 16

__global__ 
void MatrixMulKernel(float *Out, float *M, float *N, int width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 计算全局行列坐标
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float Pvalue = 0;

    // 循环遍历所有 Tile
    for (int ph = 0; ph < width / TILE_WIDTH; ++ph) {
        // 2. 修正：使用 ty, tx 填充共享内存，并修正索引计算
        // Mds 存储 M 的子块：行索引为 row，列索引为 ph*TILE + tx
        Mds[ty][tx] = M[row * width + (ph * TILE_WIDTH + tx)];
        
        // Nds 存储 N 的子块：行索引为 ph*TILE + ty，列索引为 col
        Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * width + col];
        
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }

    Out[row * width + col] = Pvalue;
}

int main() {
    int width = 2048;
    int num_elements = width * width;
    size_t num_size = sizeof(float) * num_elements;

    // 使用 vector 初始化主机内存
    vector<float> M_h(num_elements);
    vector<float> N_h(num_elements);
    vector<float> Out_h(num_elements);

    for (int i = 0; i < num_elements; ++i) {
        M_h[i] = rand() % 10;
        N_h[i] = rand() % 10;
    }

    float *M_d, *N_d, *Out_d;
    cudaMalloc((void **)&M_d, num_size);
    cudaMalloc((void **)&N_d, num_size);
    cudaMalloc((void **)&Out_d, num_size);

    // 3. 修正：cudaMemcpy 参数顺序 (dst, src)
    cudaMemcpy(M_d, M_h.data(), num_size, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h.data(), num_size, cudaMemcpyHostToDevice);

    // 4. 修正：Grid 大小应覆盖整个矩阵
    dim3 blockSize(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridSize(width / TILE_WIDTH, width / TILE_WIDTH, 1);

    cout << "Grid Size: " << gridSize.x << " x " << gridSize.y << endl;

    MatrixMulKernel<<<gridSize, blockSize>>>(Out_d, M_d, N_d, width);
    
    // 添加错误检查是个好习惯
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    
    // 同步设备，确保计算完成
    cudaDeviceSynchronize();

    cout << "Finish calculation" << endl;

    // 验证部分 (CPU计算 Out[0][0])
    float res = 0;
    for (int i = 0; i < width; ++i) {
        res += M_h[i] * N_h[i * width + 0]; // Row 0 of M * Col 0 of N
    }
    cout << "CPU 验证结果 (Out[0]): " << res << endl;

    // 拷贝回结果
    cudaMemcpy(Out_h.data(), Out_d, num_size, cudaMemcpyDeviceToHost);
    cout << "GPU 计算结果 (Out[0]): " << Out_h[0] << endl;

    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(Out_d);
    
    return 0;
}