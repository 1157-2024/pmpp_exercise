#include<iostream>
#include<cuda_runtime.h>
#include<vector>

using namespace std;
#define TILE_WIDTH 16

__global__
void MatrixKernel(float *Out,float *M,float *N,int width){
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = (bx * TILE_WIDTH) + tx;
    int row = (by * TILE_WIDTH) + ty;

    float OutValue = 0;
    for (int ph = 0; ph < (width + TILE_WIDTH - 1) / TILE_WIDTH;++ph){
        if(row<width && (ph*TILE_WIDTH+tx)<width){
            Mds[ty][tx] = M[row * width + ph * TILE_WIDTH + tx];
        }
        else
            Mds[ty][tx] = 0.0;
        if (col < width && (ph * TILE_WIDTH + ty) < width)
        {
            Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * width + col];
        }
        else
            Nds[ty][tx] = 0.0;
        __syncthreads();

        for (int i = 0; i < TILE_WIDTH;++i){
            OutValue += Mds[ty][i] * Nds[i][tx];
        }

        __syncthreads();
    }
    if(row<width && col<width){
        Out[row * width + col] = OutValue;
    }
}

int main(){
    int width = 2048;
    int num_elements = width * width;
    size_t num_size = num_elements * sizeof(float);

    vector<float> M_h(num_elements);
    vector<float> N_h(num_elements);
    vector<float> Out_h(num_elements);

    for (int i = 0; i < num_elements;++i){
        M_h[i] = rand() % 10;
        N_h[i] = rand() % 10;
    }

    float *M_d, *N_d, *Out_d;
    cudaMalloc((void **)&M_d, num_size);
    cudaMalloc((void **)&N_d, num_size);
    cudaMalloc((void **)&Out_d, num_size);

    cudaMemcpy(M_d, M_h.data(), num_size, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h.data(), num_size, cudaMemcpyHostToDevice);

    dim3 blockSize(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridSize(
        (width + TILE_WIDTH - 1) / TILE_WIDTH,
        (width + TILE_WIDTH - 1) / TILE_WIDTH,
        1
    );
    MatrixKernel<<<gridSize, blockSize>>>(Out_d, M_d, N_d, width);

    cudaMemcpy(Out_h.data(), Out_d, num_size, cudaMemcpyDeviceToHost);

    float num_host = 0;
    for (int i = 0; i < width; ++i)
    {
        num_host += M_h[i] * N_h[i * width];
    }

    cout << "Host Res = " << num_host << endl;

    cout << "Device Res = " << Out_h[0] << endl;

    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(Out_d);

    return 0;
}