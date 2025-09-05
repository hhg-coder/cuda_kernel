// #include <iostream>
// #include <math.h>
//
// // Kernel function to add the elements of two arrays
// __global__
// void add(int n, float *x, float *y)
// {
//     for (int i = 0; i < n; i++)
//         y[i] = x[i] + y[i];
// }
//
// int main(void)
// {
//     int N = 1<<20;
//     float *x, *y;
//
//     // Allocate Unified Memory – accessible from CPU or GPU
//     cudaMallocManaged(&x, N*sizeof(float));
//     cudaMallocManaged(&y, N*sizeof(float));
//
//     // initialize x and y arrays on the host
//     for (int i = 0; i < N; i++) {
//         x[i] = 1.0f;
//         y[i] = 2.0f;
//     }
//
//     // Run kernel on 1M elements on the GPU
//     add<<<1, 1>>>(N, x, y);
//
//     // Wait for GPU to finish before accessing on host
//     cudaDeviceSynchronize();
//
//     // Check for errors (all values should be 3.0f)
//     float maxError = 0.0f;
//     for (int i = 0; i < N; i++) {
//         maxError = fmax(maxError, fabs(y[i]-3.0f));
//     }
//     std::cout << "Max error: " << maxError << std::endl;
//
//     // Free memory
//     cudaFree(x);
//     cudaFree(y);
//     return 0;
// }

// #include <stdio.h>
#include "common.cuh"

void initialData(float *addr, int size) {
    for (int i = 0; i < size; i++) {
        addr[i] = static_cast<float>(rand() & 0xff) / 10.f;
    }
}

__device__ float add(float a, float b) {
    return a + b;
}


__global__ void addFromGPU(float *dataA, float *dataB, float *dataC, const int size) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int id = bid * blockDim.x + tid;

    if (id >= size) {return;}
    dataC[id] = add(dataA[id], dataB[id]);
}

int main() {
    //1 设置GPU
    setGPU();
    //2 分配主机和设备内存
    constexpr int iElemCount = 512;
    constexpr size_t stBytes = iElemCount * sizeof(float);
    //分配主机内存
    float *fpHostData_A = static_cast<float *>(malloc(stBytes));
    float *fpHostData_B = static_cast<float *>(malloc(stBytes));
    float *fpHostData_C = static_cast<float *>(malloc(stBytes));
    if (fpHostData_A != nullptr && fpHostData_B != nullptr && fpHostData_C != nullptr) {
        memset(fpHostData_A, 0, stBytes);
        memset(fpHostData_B, 0, stBytes);
        memset(fpHostData_C, 0, stBytes);
    }
    else {
        printf("Host Memory allocation error\n");
        exit(-1);
    }

    //分配设备内存
    float *fpDevicData_A, *fpDevicData_B, *fpDevicData_C;
    cudaMalloc(&fpDevicData_A, stBytes);
    cudaMalloc(&fpDevicData_B, stBytes);
    cudaMalloc(&fpDevicData_C, stBytes);
    if (fpHostData_A != nullptr && fpHostData_B != nullptr && fpHostData_C != nullptr) {
        cudaMemset(fpHostData_A, 0, stBytes);
        cudaMemset(fpHostData_B, 0, stBytes);
        cudaMemset(fpHostData_C, 0, stBytes);
    }
    else {
        printf("Device Memory allocation error\n");
        free(fpHostData_A);
        free(fpHostData_B);
        free(fpHostData_C);
        exit(-1);
    }

    //3 初始化
    srand(666);
    initialData(fpHostData_A, iElemCount);
    initialData(fpHostData_B, iElemCount);
    //4 数据从主机复制到设备
    cudaMemcpy(fpDevicData_A, fpHostData_A, stBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(fpDevicData_B, fpHostData_B, stBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(fpDevicData_C, fpHostData_C, stBytes, cudaMemcpyHostToDevice);

    //5 调用kernel函数进行计算
    dim3 dimGrid(iElemCount / 32, 1, 1);
    dim3 dimBlock(32, 1, 1);
    addFromGPU<<<dimGrid, dimBlock>>>(fpDevicData_A, fpDevicData_B, fpDevicData_C, iElemCount);
    cudaDeviceSynchronize();//join 等待同步 汇入主线程

    //6 将计算得到的数据传给主线程
    cudaMemcpy(fpHostData_C, fpDevicData_C, stBytes, cudaMemcpyDeviceToHost);

    // 打印
    for (int i = 0; i < 10; i++) {
        printf("id: %d\tmatrix_A: %.2f\tmatrix_B: %.2f\tmatrix_C: %.2f\n",i + 1, \
            fpHostData_A[i], fpHostData_B[i], fpHostData_C[i]);
    }
    cudaFree(fpDevicData_A);
    cudaFree(fpDevicData_B);
    cudaFree(fpDevicData_C);
    free(fpHostData_A);
    free(fpHostData_B);
    free(fpHostData_C);

    cudaDeviceReset();
    return 0;
}