#include<cuda_runtime.h>
#include<iostream>

using namespace std;

int main(){
    // 1. 获取设备数量
    int devCount;
    cudaGetDeviceCount(&devCount);

    cout << "发现 " << devCount << " 个 CUDA 设备。" << endl;

    // 2. 遍历每个设备查询属性
    cudaDeviceProp devProp;
    for(unsigned int i = 0; i < devCount; i++) {
        cudaGetDeviceProperties(&devProp, i);

        cout << "------------------------------------------------" << endl;
        cout << "设备 " << i << ": " << devProp.name << endl; // 打印显卡名字
        
        
        // 计算能力 (Compute Capability)
        cout << "  计算能力: " << devProp.major << "." << devProp.minor << endl;
        
        // SM 数量 (书里提到的 multiProcessorCount)
        cout << "  SM 数量: " << devProp.multiProcessorCount << endl;
        
        // 每个 Block 的最大线程数 (maxThreadsPerBlock)
        cout << "  每个 Block 最大线程数: " << devProp.maxThreadsPerBlock << endl;
                
        // Warp 大小
        cout << "  Warp 大小: " << devProp.warpSize << endl;
        
        // 显存大小 (Global Memory)
        cout << "  显存总量: " << devProp.totalGlobalMem / (1024.0 * 1024.0) << " MB" << endl;
        
        cout << "------------------------------------------------" << endl;
        

    }

    return 0;
}
