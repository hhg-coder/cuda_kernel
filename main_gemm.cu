#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(float_var) (reinterpret_cast<float4*>(&(float_var))[0])
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            printf("CUDA Error: \n");                                          \
            printf("    File:       %s\n", __FILE__);                          \
            printf("    Line:       %d\n", __LINE__);                          \
            printf("    Error Code: %d\n", err);                               \
            printf("    Error Text: %s\n", cudaGetErrorString(err));           \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

void gemm_cpu(float* A, float* B, float* C, int M, int N, int K){
    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++){
            for (int k = 0; k < K; k++){
                C[i * N + j] += A[i * K + k] * B[k * N + j]; 
            }
        }
    }
}

void init_gemm_random(float* A, int M, int N){
    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++){
            A[i * N + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
}

void __global__ gemm_kernel(float *A, float *B, float *C, int M, int N, int K){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < M && col < N){
        float sum = 0.0f;
        for (int k = 0; k < K; k++){
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void launch_kernel_1(){
    int M = 1024;
    int N = 1024;
    int K = 1024;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    float* h_A = (float*)malloc(size_A);
    float* h_B = (float*)malloc(size_B);
    float* h_C = (float*)malloc(size_C);

    init_gemm_random(h_A, M, K);
    init_gemm_random(h_B, K, N);

    float *d_A , *d_B, *d_C;
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    int TN = 16;
    int TM = 32;
    dim3 blockSize(TN, TM);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
    
    //kernel 计时
    int warm_up = 2;
    int repeat = 10;
    for (int i = 0; i < warm_up; i++){
        gemm_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();
    }

    float kernel_time_sum = 0.0f;
    for (int i = 0; i < repeat; i++){
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        gemm_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float kernel_time = 0.0f;
        cudaEventElapsedTime(&kernel_time, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        kernel_time_sum += kernel_time;
    }
    printf("CUDA kernel time is : %.3f ms\n", kernel_time_sum / repeat);
    

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    float *h_C_cpu = (float*)malloc(size_C);

    //CPU计时
    auto cpu_start = std::chrono::high_resolution_clock::now();
    gemm_cpu(h_A, h_B, h_C_cpu, M, N, K);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    printf("CPU gemm time: %.3f ms\n", cpu_time);
    
    float max_diff = 0.0f;
    for (int i = 0; i < M * N; i++){
        if (std::fabs(h_C[i] - h_C_cpu[i]) > max_diff){
            max_diff = std::fabs(h_C[i] - h_C_cpu[i]);
        }
    }
    printf("max diff is : %f\n", max_diff);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_cpu);
}

__global__ void sgemm_gpu_kernel_v2(float *__restrict__ A,
                                    float *__restrict__ B,
                                    float *__restrict__ C, const int M,
                                    const int N, const int K) {
    const int BM = 16, BN = 16;
    const int BK = 64;
    __shared__ float s_a[BM][BK], s_b[BK][BN];
    float c = 0.0f;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    // 每次从全局内存加载到共享内存，每个线程都负责一个float4。以下是当前线程负责的这个float4的索引
    const int row_s_a = tid / 16;
    const int col_s_a = (tid % 16) * 4;
    const int row_s_b = tid / 4;
    const int col_s_b = (tid % 4) * 4;
    // 每个线程从读取的全局内存的位置，在A上的行是固定不变的，在B上列是固定不变的
    const int row_A = blockIdx.y * BM + row_s_a;
    const int col_B = blockIdx.x * BN + col_s_b;

    for (int step = 0; step < K / BK; step++) {
        // 从A加载到s_a
        const int col_A = step * BK + col_s_a;
        const int index_A = OFFSET(row_A, col_A, K);
        FETCH_FLOAT4(s_a[row_s_a][col_s_a]) = FETCH_FLOAT4(A[index_A]);
        // 从B加载到s_b
        const int row_B = step * BK + row_s_b;
        const int index_B = OFFSET(row_B, col_B, N);
        FETCH_FLOAT4(s_b[row_s_b][col_s_b]) = FETCH_FLOAT4(B[index_B]);
        __syncthreads();
        // 计算
        for (int k = 0; k < BK; k++) {
            const float a = s_a[threadIdx.y][k];
            const float b = s_b[k][threadIdx.x];
            c += a * b;
        }
        __syncthreads();
    }
    // 写入C
    const int row_C = blockIdx.y * BM + threadIdx.y;
    const int col_C = blockIdx.x * BN + threadIdx.x;
    const int index_C = OFFSET(row_C, col_C, N);
    C[index_C] = c;
}

void __global__ gemm_kernel_fix_2(float* A, float* B, float* C, 
    const int M, const int N, const int K){
    const int BM = 16;
    const int BN = 16;
    const int BK = 64;
    __shared__ float sm_a[BM][BK];
    __shared__ float sm_b[BK][BN];

    int row = blockIdx.y * BM + threadIdx.y;
    int col = blockIdx.x * BN + threadIdx.x;
    float c = 0.0f;

    for (int step = 0; step < K / BK; ++step) {
        // load A block to shared memory
        for (int k = threadIdx.x; k < BK; k += BN) {
            if (row < M && (step * BK + k) < K)
                sm_a[threadIdx.y][k] = A[row * K + step * BK + k];
            else
                sm_a[threadIdx.y][k] = 0.0f;
        }
        // load B block to shared memory
        for (int k = threadIdx.y; k < BK; k += BM) {
            if ((step * BK + k) < K && col < N)
                sm_b[k][threadIdx.x] = B[(step * BK + k) * N + col];
            else
                sm_b[k][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        for (int k = 0; k < BK; ++k) {
            c += sm_a[threadIdx.y][k] * sm_b[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = c;

}

void __global__ gemm_kernel_2(float* A, float* B, float* C, 
    const int M, const int N, const int K){
    const int BM = 16;
    const int BN = 16;
    const int BK = 64;
    __shared__ float sm_a[BM][BK];
    __shared__ float sm_b[BK][BN];
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int row_s_a = tid / 16;
    const int col_s_a = (tid % 16) * 4;
    const int row_s_b = tid / 4;
    const int col_s_b = (tid % 4) * 4;
    // int row_s_a = threadIdx.y;
    // int col_s_a = threadIdx.x * 4;
    // int row_s_b = threadIdx.y * 4;
    // int col_s_b = (threadIdx.x % 4) * 4;

    int row_A = row_s_a + BM * blockIdx.y;
    int col_B = col_s_b + BN * blockIdx.x;

    int row_C = row_A;
    int col_C = col_B;
    float c = 0.0f;

    for (int step = 0; step < (K / BK); step++){
        int col_A = step * BK + col_s_a;
        int row_B = step * BK + row_s_b;

        FETCH_FLOAT4(sm_a[row_s_a][col_s_a]) = FETCH_FLOAT4(A[row_A * K + col_A]);
        FETCH_FLOAT4(sm_b[row_s_b][col_s_b]) = FETCH_FLOAT4(B[row_B * N + col_B]);
        __syncthreads();
        for (int k = 0; k < BK; k++){
            c += sm_a[row_s_a][k] * sm_b[k][col_s_b];
        }
    }
    C[row_C * N + col_C] = c;

}

void launch_kernel_2(){
    int M = 1024;
    int N = 1024;
    int K = 1024;
    int BM = 16;
    int BN = 16;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    float* h_A = (float*)malloc(size_A);
    float* h_B = (float*)malloc(size_B);
    float* h_C = (float*)malloc(size_C);

    init_gemm_random(h_A, M, K);
    init_gemm_random(h_B, K, N);

    float *d_A , *d_B, *d_C;
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    
    dim3 blockSize(BN, BM);
    dim3 gridSize((N + BN - 1) / BN, (M + BM - 1) / BM);

    int warm_up = 2;
    int repeat = 5;
    for (int i = 0; i < warm_up; i++){
        gemm_kernel_fix_2<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    float kernel_time_sum = 0.0f;
    for (int i = 0; i < repeat; i++){
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        gemm_kernel_fix_2<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float kernel_time = 0.0f;
        cudaEventElapsedTime(&kernel_time, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        kernel_time_sum += kernel_time;
    }
    printf("kernel_2_time_sum is : %.3fms\n", kernel_time_sum / repeat);
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    float *h_C_cpu = (float*)malloc(size_C);

    //CPU计时
    auto cpu_start = std::chrono::high_resolution_clock::now();
    gemm_cpu(h_A, h_B, h_C_cpu, M, N, K);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    printf("CPU gemm time: %.3f ms\n", cpu_time);
    
    float max_diff = 0.0f;
    for (int i = 0; i < M * N; i++){
        if (std::fabs(h_C[i] - h_C_cpu[i]) > max_diff){
            max_diff = std::fabs(h_C[i] - h_C_cpu[i]);
        }
    }
    printf("max diff is : %f\n", max_diff);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_cpu);
    
}

int main(){
    launch_kernel_1();
    printf("=============\n");
    launch_kernel_2();
    return 0;
}