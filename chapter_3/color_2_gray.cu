#include<cuda_runtime.h>
#include<iostream>
#include<vector>

using namespace std;

__global__
void color_2_gray(unsigned char* Pout_d,unsigned char*Pin_d,int width, int height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x<width && y<height){
        int gray_index = y * width + x;
        int rgb_index = 3*gray_index;
        unsigned char r = Pin_d[rgb_index];
        unsigned char g = Pin_d[rgb_index + 1];
        unsigned char b = Pin_d[rgb_index + 2];

        Pout_d[gray_index] = 0.21f * r + 0.72f * g + 0.07f * b;
    }
}

int main(){
    // 模拟书上图片
    int width = 62;
    int height = 76;
    size_t gray_size = sizeof(char) * width * height;
    size_t rgb_size = gray_size * 3;

    vector<unsigned char> Pin_h(rgb_size);
    vector<unsigned char> Pout_h(gray_size);

    // 假设图片是全红
    for (int i = 0; i < width*height;i++){
        Pin_h[i * 3] = 255;
        Pin_h[i * 3 + 1] = 0;
        Pin_h[i * 3 + 2] = 0;
    }

    // 申请设备显存
    unsigned char *Pin_d,*Pout_d;
    cudaMalloc(&Pin_d, rgb_size);
    cudaMalloc(&Pout_d, gray_size);

    // host 2 device
    cudaMemcpy(Pin_d, Pin_h.data(), rgb_size, cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16);
    dim3 dimGrid(
        (width + dimBlock.x + 1) / dimBlock.x,
        (height + dimBlock.y - 1) / dimBlock.y);

    color_2_gray<<<dimGrid, dimBlock>>>(Pout_d, Pin_d, width, height);

    cudaDeviceSynchronize();

    // device 2 host
    cudaMemcpy(Pout_h.data(), Pout_d, gray_size, cudaMemcpyDeviceToHost);

    // 验证一下
    cout << "Pixel 0 Gray Value: " << (int)Pout_h[0] << endl;
    if(Pout_h[0] == 53) {
        cout << "Test Passed!" << endl;
    }

    // free
    cudaFree(Pin_d);
    cudaFree(Pout_d);

    return 0;
}