#include<cuda_runtime.h>
#include<vector>
#include<iostream>

using namespace std;

#define BLUR_SIZE 1

__global__
void blurKernel(unsigned char *Pout_d, unsigned char *Pin_d,int w,int h){
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    if(col < w && row<h){
        int PixelVals = 0;
        int PixelNums = 0;
        for (int del_row = -BLUR_SIZE; del_row <= BLUR_SIZE;del_row++){
            for (int del_col = -BLUR_SIZE; del_col <= BLUR_SIZE;del_col++){
                int cur_col = col + del_col;
                int cur_row = row + del_row;

                if(cur_col>=0 && cur_col<w && cur_row>=0 && cur_row<h){
                    PixelVals += Pin_d[cur_row * w + cur_col];
                    PixelNums++;
                }
            }
        }

        Pout_d[row * w + col] = (unsigned char)(PixelVals / PixelNums);
    }
}

int main(){
    // 造图片数据
    int height = 1024;
    int width = 2048;
    size_t gray_size = height * width * sizeof(char);

    vector<unsigned char> Pin_h(gray_size);
    vector<unsigned char> Pout_h(gray_size);
    for (int i = 0; i < height*width; i++){
        Pin_h[i] = rand() % 256;
    }

    //申请显存空间
    unsigned char *Pin_d, *Pout_d;
    cudaMalloc((void **)&Pin_d, gray_size);
    cudaMalloc((void **)&Pout_d, gray_size);

    // 搬数据
    cudaMemcpy(Pin_d, Pin_h.data(), gray_size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y);

    blurKernel<<<gridSize, blockSize>>>(Pout_d, Pin_d, width, height);
    cudaError_t err = cudaGetLastError();
    if(err!=cudaSuccess){
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        // 如果出错，直接退出，方便你调试
        return -1;
    }

    cudaMemcpy(Pout_h.data(), Pout_d, gray_size, cudaMemcpyDeviceToHost);

    // 检查一下
    cout << "Pin[0][0]~Pin[2][2]" << endl;
    int sum = 0;
    for (int i = -BLUR_SIZE; i <= BLUR_SIZE; i++)
    {
        for (int j = -BLUR_SIZE; j <= BLUR_SIZE;j++){
            int cur_row = i + 1;
            int cur_col = j + 1;
            cout << int(Pin_h[cur_row*width + cur_col]) << " ";
            sum += Pin_h[cur_row * width + cur_col];
        }
        cout << endl;
    }
    cout << "avg = " << sum / 9 << endl;
    cout << "Pout[1][1]: " << int(Pout_h[1 * width + 1]) << endl;
    cudaFree(Pin_d);
    cudaFree(Pout_d);
    return 0;
}