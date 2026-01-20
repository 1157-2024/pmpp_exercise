#include<cuda_runtime.h>
#include<iostream>
#include<vector>

using namespace std;

__global__
void matrix_mul(float *Out_d,float *M_d, float *N_d, int width){
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if(col<width && row<width){
        float res_sum = 0;
        for (int i = 0; i < width; i++)
        {
            res_sum += M_d[row * width + i] * N_d[i * width + col];
        }
        Out_d[row * width + col] = res_sum;
    }
}

int main(){
    // 初始化数据
    int width = 100;
    size_t size = width * width * sizeof(float);
    vector<float> M_h(size);
    vector<float> N_h(size);
    vector<float> Out_h(size);

    for (int i = 0; i < size;i++){
        M_h[i] = rand() % 10;
        N_h[i] = rand() % 10;
    }
    
    // 申请显存
    float *M_d, *N_d, *Out_d;
    cudaMalloc((void **)&M_d, size);
    cudaMalloc((void **)&N_d, size);
    cudaMalloc((void **)&Out_d, size);

    cudaMemcpy(M_d, M_h.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h.data(), size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (width + blockSize.y - 1) / blockSize.y
    );
    matrix_mul<<<gridSize, blockSize>>>(Out_d, M_d, N_d, width);

    cudaMemcpy(Out_h.data(), Out_d, size, cudaMemcpyDeviceToHost);

    // 验证一下
    cout <<"Out[0] = " <<Out_h[0] << endl;
    float res = 0;
    for (int i = 0; i < width; i++)
    {
        res += M_h[i] * N_h[i * width];
    }
    cout << res << endl;

    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(Out_d);

    return 0;
}