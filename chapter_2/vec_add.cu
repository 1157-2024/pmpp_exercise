#include<stdio.h>
#include<cuda_runtime.h>
#include<malloc.h>
#include<iostream>

using namespace std;

__global__ void vec_add_kernel(float *A_d,float *B_d, float *C_d,int n){
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if(i<n){
        A_d[i] = B_d[i] + C_d[i];
    }
}

void vec_add(float *A_h,float *B_h, float *C_h, int n){
    //申请显存空间
    float *A_d, *B_d,*C_d;
    cudaMalloc((void **)&A_d, sizeof(float) * n);
    cudaMalloc((void **)&B_d, sizeof(float) * n);
    cudaMalloc((void **)&C_d, sizeof(float) * n);

    //把内存搬去显存
    cudaMemcpy(B_d, B_h, sizeof(float) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C_h, sizeof(float) * n, cudaMemcpyHostToDevice);

    //调用核函数
    int blockSize = 256;
    // (n + 256 - 1) / 256 确保向上取整
    int gridSize = (n + blockSize - 1) / blockSize;
    vec_add_kernel<<<gridSize, blockSize>>>(A_d, B_d, C_d, n);

    //把运算结果搬回来
    cudaMemcpy(A_h, A_d, sizeof(float) * n, cudaMemcpyDeviceToHost);
    // 记得释放显存
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main(){
    // 主机参数申请空间
    float *A_h,*B_h,*C_h;
    int n = 1000;
    int size_float = sizeof(float);
    A_h = (float*)malloc(size_float * n);
    B_h = (float*)malloc(size_float * n);
    C_h = (float*)malloc(size_float * n);

    for (int i = 0; i < n; i++) {
        B_h[i] = i;       
        C_h[i] = i * 2;   
    }

    vec_add(A_h, B_h, C_h,n);

    // 验证前10个数
    for (int i = 0; i < 10; i++) {
        cout << i << ": " << B_h[i] << " + " << C_h[i] << " = " << A_h[i] << endl;
    }

    free(A_h);
    free(B_h);
    free(C_h);
}