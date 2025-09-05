#define CEIL(a,b) (((a)+(b)-1)/(b))
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value)[0])
// int block_size = 1024;
// int grid_size = (N + 1024 - 1) / 1024;

// elementwise_add<<<grid_size, block_size>>>(a, b, c, N);
__global__ void elementwise_add(float * a, float * b, float * c, int N) {
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    if (idx < N) {
        float4 temp_a = FLOAT4(a[idx]);
        float4 temp_b = FLOAT4(b[idx]);
        float4 temp_c;
        temp_c.x = temp_a.x + temp_b.x;
        temp_c.y = temp_a.y + temp_b.y;
        temp_c.z = temp_a.z + temp_b.z;
        temp_c.w = temp_a.w + temp_b.w;
        FLOAT4(c[idx]) = temp_c;
    }
}

#define THREAD_PER_BLOCK 256
#define WARP_SIZE 32
template <unsigned int blockSize>
__device__ __forceinline__ float warpReduceSum(float sum) {
    if (blockSize >= 32)sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
    if (blockSize >= 16)sum += __shfl_down_sync(0xffffffff, sum, 8);// 0-8, 1-9, 2-10, etc.
    if (blockSize >= 8)sum += __shfl_down_sync(0xffffffff, sum, 4);// 0-4, 1-5, 2-6, etc.
    if (blockSize >= 4)sum += __shfl_down_sync(0xffffffff, sum, 2);// 0-2, 1-3, 4-6, 5-7, etc.
    if (blockSize >= 2)sum += __shfl_down_sync(0xffffffff, sum, 1);// 0-1, 2-3, 4-5, etc.
    return sum;
}
//最后输出blocksize大小的output
template <unsigned int blockSize, int NUM_PER_THREAD>
__global__ void reduce_warp_shuffle(float* input, float* output, int N) {
    int sum = 0;
    int tid = threadIdx.x;
    int i = blockIdx.x * (blockSize * NUM_PER_THREAD) + tid;
    for (int j = 0; j < blockSize; j++) {
        if (i + j * NUM_PER_THREAD < N) {
            sum += input[i + j * NUM_PER_THREAD];
        }
    }
    __shared__ float warpSums[WARP_SIZE]; // 每个warp的部分和
    sum = warpReduceSum<blockSize>(sum); // 每个warp内部归约
    int laneid = tid % WARP_SIZE; // warp内线程id
    int warpid = tid / WARP_SIZE; // warp id
    if (laneid == 0) {
        warpSums[warpid] = sum; // 每个warp的第一个线程写入部分和
    }
    __syncthreads();
    // 让第一个warp的线程来归约warpSums
    sum = (tid < (blockSize / WARP_SIZE)) ? warpSums[laneid] : 0;
    if (warpid == 0) {
        sum = warpReduceSum<blockSize / WARP_SIZE>(sum);
    }
    if (tid == 0) {
        output[blockIdx.x] = sum; // 每个block的第一个线程写入最终结果
    }
}

__global__ void softmax_online(float* input, float* output, int N) {
    __shared__ float sdata[THREAD_PER_BLOCK]; // 每个block的共享内存
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    float max_val = -FLT_MAX;
    // 1. 计算局部最大值
    if (i < N) {
        max_val = input[i];
    }
    sdata[tid] = max_val;
    __syncthreads();
    // 归约计算block内最大值
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    max_val = sdata[0]; // block内最大值
    __syncthreads();
    // 2. 计算指数和
    float sum_exp = 0.0f;
    if (i < N) {
        sum_exp = expf(input[i] - max_val);
    }
    sdata[tid] = sum_exp;
    __syncthreads();
    // 归约计算block内指数和
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float total_sum_exp = sdata[0]; // block内指数和
    __syncthreads();
    // 3. 计算softmax输出
    if (i < N) {
        output[i] = expf(input[i] - max_val) / total_sum_exp;
    }
}   




