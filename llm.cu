#include <iostream>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

#define THREAD_PER_BLOCK 256
#define ARRAY_SIZE (1 << 20) // 1M elements


//有线束分化
__global__ void reduce0(float *d_in,float *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];

    //each thread loads one element from global memory to shared mem
    unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int tid=threadIdx.x;
    sdata[tid]=d_in[i];
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=1; s<blockDim.x; s*=2){
        if(tid%(2*s) == 0){
            sdata[tid]+=sdata[tid+s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if(tid==0)d_out[blockIdx.x]=sdata[tid];
}

//有bank冲突
__global__ void reduce1(float *d_in,float *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];

    //each thread loads one element from global memory to shared mem
    unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int tid=threadIdx.x;
    sdata[tid]=d_in[i];
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=1; s<blockDim.x; s*=2){
        int index = 2*s*tid;
        if(index < blockDim.x){
            sdata[index]+=sdata[index+s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if(tid==0)d_out[blockIdx.x]=sdata[tid];
}

// 有一半线程一次加法也没做
__global__ void reduce2(float *d_in, float *d_out) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 将数据加载到共享内存
    sdata[tid] = d_in[i];
    __syncthreads();

    // 树形归约
    for(unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 块结果写入全局内存
    if (tid == 0) d_out[blockIdx.x] = sdata[0];
}


//利用闲置线程
__global__ void reduce3(float *d_in,float *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];

    //each thread loads one element from global memory to shared mem
    unsigned int i=blockIdx.x*(blockDim.x*2)+threadIdx.x;
    unsigned int tid=threadIdx.x;
    sdata[tid]=d_in[i] + d_in[i+blockDim.x];
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=blockDim.x/2; s>0; s>>=1){
        if(tid < s){
            sdata[tid]+=sdata[tid+s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if(tid==0)d_out[blockIdx.x]=sdata[tid];
}

const int WARP_SIZE = 32;
template <unsigned int blockSize>
__device__ __forceinline__ float warpReduceSum(float sum){
    if(blockSize >= 32)sum += __shfl_down_sync(0xffffffff,sum,16);
    if(blockSize >= 16)sum += __shfl_down_sync(0xffffffff,sum,8);
    if(blockSize >= 8)sum += __shfl_down_sync(0xffffffff,sum,4);
    if(blockSize >= 4)sum += __shfl_down_sync(0xffffffff,sum,2);
    if(blockSize >= 2)sum += __shfl_down_sync(0xffffffff,sum,1);
    return sum;
}


//shuffle指令
template <unsigned int blockSize>
__global__ void reduce7(float *d_in,float *d_out, unsigned int n){
    float sum = 0;

    //each thread loads one element from global memory to shared mem
    unsigned int i=blockIdx.x*(blockDim.x*2)+threadIdx.x;
    unsigned int tid=threadIdx.x;
    unsigned int gridSize = blockSize * 2 * gridDim.x;

    while(i<n){
        sum +=d_in[i]+d_in[i+blockSize];
        i+=gridSize;
    }

    // shared mem for partial sums(one per warp in the block
    static __shared__ float warpLevelSums[WARP_SIZE];
    const int laneId = threadIdx.x % WARP_SIZE;
    const int warpId = threadIdx.x / WARP_SIZE;

    sum = warpReduceSum<blockSize>(sum);

    if(laneId == 0)warpLevelSums[warpId]=sum;
    __syncthreads();

    sum = (threadIdx.x < blockDim.x / WARP_SIZE)? warpLevelSums[laneId]:0;
    // Final reduce using first warp
    if(warpId == 0)sum = warpReduceSum<blockSize/WARP_SIZE>(sum);
    // write result for this block to global mem
    if(tid==0)d_out[blockIdx.x]=sum;
}

int main() {
    const int N = ARRAY_SIZE;
    int block_size = THREAD_PER_BLOCK;
    const int grid_size = (N + block_size - 1) / block_size;
    block_size /= 2;

    // 主机内存分配
    float *h_in = new float[N];
    float *h_out = new float[grid_size];
    float gpu_result = 0.0f, cpu_result = 0.0f;

    // 初始化输入数据
    for(int i = 0; i < N; i++) {
        h_in[i] = static_cast<float>(rand()) / RAND_MAX; // 0~1之间的随机数
    }

    // GPU计算计时开始
    auto start_gpu = std::chrono::high_resolution_clock::now();

    // 设备内存分配
    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, grid_size * sizeof(float));

    // 数据复制到设备
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    // 启动核函数
    reduce3<<<grid_size, block_size>>>(d_in, d_out);

    // 数据复制回主机
    cudaMemcpy(h_out, d_out, grid_size * sizeof(float), cudaMemcpyDeviceToHost);

    // 主机归约最终结果
    for(int i = 0; i < grid_size; i++) {
        gpu_result += h_out[i];
    }

    // GPU计算计时结束
    auto end_gpu = std::chrono::high_resolution_clock::now();
    auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu).count();

    // CPU计算计时开始
    auto start_cpu = std::chrono::high_resolution_clock::now();

    // CPU串行归约
    for(int i = 0; i < N; i++) {
        cpu_result += h_in[i];
    }

    // CPU计算计时结束
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count();

    // 输出结果
    std::cout << "GPU Result: " << gpu_result << " | Time: " << gpu_time << " μs\n";
    std::cout << "CPU Result: " << cpu_result << " | Time: " << cpu_time << " μs\n";
    std::cout << "Speedup: " << static_cast<double>(cpu_time) / gpu_time << "x\n";
    std::cout << "Difference: " << std::abs(gpu_result - cpu_result) << " ("
              << (std::abs(gpu_result - cpu_result) / cpu_result * 100.0) << "%)\n";

    // 释放资源
    delete[] h_in;
    delete[] h_out;
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}