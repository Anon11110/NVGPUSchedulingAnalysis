#include <chrono>
#include "../include/KernelGenerator.hpp"

KernelGenerator::KernelGenerator(int num_streams, int max_threads, int max_blocks, int max_shared_mem)
    : max_threads_(max_threads), max_blocks_(max_blocks), max_shared_mem_(max_shared_mem),
    rng_(std::chrono::steady_clock::now().time_since_epoch().count())
{
    for (int i = 0; i < num_streams; i++) {
        stream_manager_.emplace_back();
    }
}


template<typename T>
T KernelGenerator::GetRandomNumber(T min, T max)
{
    std::uniform_int_distribution<T> dist(min, max);
    return dist(rng_);
}

void KernelGenerator::GenerateAndLaunchKernels(int num_kernels, const std::vector<KernelSetting>& settings)
{
    for (int i = 0; i < num_kernels; i++) {
        int stream_index = i % stream_manager_.size();
        KernelSetting setting = i < settings.size() ? settings[i] : KernelSetting();

        int thread_per_block = setting.threads_per_block.value_or(GetRandomNumber(1, max_threads_));
        int blocks = setting.blocks.value_or(GetRandomNumber(1, max_blocks_));
        int shared_mem_size = setting.shared_mem_size.value_or(GetRandomNumber(0, max_shared_mem_));

        // Allocate memory for kernel parameters
        int* d_smids, *d_block_ids, *d_thread_ids, *d_block_dims, *d_thread_dims, *d_shared_mem_sizes;
        float* d_kernel_durations;
        cudaMalloc(&d_smids, blocks * sizeof(int));
        cudaMalloc(&d_block_ids, blocks * sizeof(int));
        cudaMalloc(&d_thread_ids, blocks * thread_per_block * sizeof(int));
        cudaMalloc(&d_block_dims, blocks * sizeof(int));
        cudaMalloc(&d_thread_dims, blocks * thread_per_block * sizeof(int));
        cudaMalloc(&d_shared_mem_sizes, blocks * sizeof(int));
        cudaMalloc(&d_kernel_durations, blocks * sizeof(float));

        std::vector<float> kernel_durations(blocks, config.kernelDuration.value_or(0.1f));
        cudaMemcpy(d_kernel_durations, kernel_durations.data(), blocks * sizeof(float), cudaMemcpyHostToDevice);

        auto kernelFunction = [d_smids, d_block_ids, d_thread_ids, d_block_dims, d_thread_dims, d_shared_mem_sizes, d_kernel_durations](cudaStream_t stream) {
            TestKernel<<<blocks, thread_per_block, shared_mem_size, stream>>>(d_smids, d_block_ids, d_thread_ids, d_block_dims, d_thread_dims, d_shared_mem_sizes, d_kernel_durations, 900);
        };

        streamManagers_[streamIndex].AddKernel("TestKernel" + std::to_string(i), dim3(blocks), dim3(threadsPerBlock), sharedMemSize, kernelFunction);
        streamManagers_[streamIndex].ScheduleKernelExecution("TestKernel" + std::to_string(i));
    }
}

__global__ void KernelGenerator::TestKernel(int *smids, int *block_ids, int *thread_ids, int *block_dims, int *thread_dims,
                           int *shared_mem_sizes, float *kernel_durations, clock_t clock_rate) {
    // Get the block ID and thread ID
    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;

    // Get the SM ID using inline assembly
    int smid;
    asm("mov.u32 %0, %%smid;" : "=r"(smid));

    // Store the SM ID, block ID, and thread ID in global memory
    if (thread_id == 0) {
        smids[block_id] = smid;
        block_ids[block_id] = block_id;
        block_dims[block_id] = blockDim.x;
        shared_mem_sizes[block_id] = (int)blockDim.x * sizeof(int);
    }
    thread_ids[block_id * blockDim.x + thread_id] = thread_id;
    thread_dims[block_id * blockDim.x + thread_id] = threadIdx.x;

    // Allocate shared memory
    extern __shared__ int shared_mem[];

    // Initialize shared memory with thread IDs
    shared_mem[thread_id] = thread_id;
    __syncthreads();

    // Perform some computations using registers and shared memory
    int reg_val = thread_id;
    for (int i = 0; i < blockDim.x; i++) {
        reg_val += shared_mem[i];
    }

    // Introduce a delay based on the kernel duration
    clock_t start_time = clock64();
    float kernel_duration = kernel_durations[block_id];
    while ((clock64() - start_time) / (clock_rate * 1e-3f) < kernel_duration) {
        // Busy wait
    }
}