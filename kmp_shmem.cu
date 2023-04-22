#include "kmp_shmem.cuh"

#include "common/utils.hpp"
#include "kmp_cpu.hpp"

// This value should be several times larger than pattern_length.
static constexpr int match_length_per_thread = 20;

static constexpr int output_cache_size_per_block = 20;

static constexpr int block_size = 16;

static __host__ __device__ int ceil_div(int x, int y) {
    return (x - 1) / y + 1;
}

static int round_up_to_multiple(int x, int y) {
    return ceil_div(x, y) * y;
}

// Get arr[idx], where arr is the compact form of the gene sequence.
static __host__ __device__ inline char get(const char *arr, int idx) {
    return (arr[idx>>2] >> ((idx & 0x3) << 1)) & 0x3;
}

__device__ void KMP_search(
    const char *text, int text_start, int text_end,
    const char *pattern, int pattern_length,
    int *shared_output, int *shared_output_cnt,
    int *global_output, int *global_output_cnt, int max_output_cnt,
    int *fail
) {
    printf("thread %d, %d: start %d, end %d\n", blockIdx.x, threadIdx.x, text_start, text_end);
    int i = text_start, j = 0;
    while (i < text_end) {
        if (get(text, i) == get(pattern, j)) {
            i++; j++;
            if (j == pattern_length) {
                // Occurrence found. Try write the output to the shared memory.
                int output_idx = atomicAdd(shared_output_cnt, 1);
                if (output_idx < output_cache_size_per_block) {
                    shared_output[output_idx] = i - j;
                }
                else {
                    // Write to global memory.
                    int output_idx = atomicAdd(global_output_cnt, 1);
                    if (output_idx < max_output_cnt) {
                        global_output[output_idx] = i - j;
                    }
                    else break;
                }
                j = fail[j];
            }
        }
        else {
            j = fail[j];
            if (j < 0) {
                i++, j++;
            }
        }
    }
}

__global__ void KMP_search_shmem_kernel(
    const char *text, int text_length, const char *pattern, int pattern_length,
    int *output, int *output_cnt, int max_output_cnt, int *fail
) {
    extern __shared__ char shared_memory[];
    int *shared_fail = reinterpret_cast<int *>(shared_memory);
    int *shared_output = reinterpret_cast<int *>(
        shared_memory
        + sizeof(int) * (pattern_length + 1)
    );
    int *shared_output_cnt = reinterpret_cast<int *>(
        shared_memory
        + sizeof(int) * (pattern_length + 1)
        + sizeof(int) * output_cache_size_per_block
    );
    int *shared_global_output_start = reinterpret_cast<int *>(
        shared_memory
        + sizeof(int) * (pattern_length + 1)
        + sizeof(int) * output_cache_size_per_block
        + sizeof(int)
    );
    char *shared_pattern = (
        shared_memory
        + sizeof(int) * (pattern_length + 1)
        + sizeof(int) * output_cache_size_per_block
        + 2 * sizeof(int)
    );
    if (threadIdx.x == 0) {
        printf("shared_pattern offset = %d\n", shared_pattern - shared_memory);
    }

    int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    int match_start = (match_length_per_thread - (pattern_length - 1)) * global_index;
    int match_end = match_start + match_length_per_thread;
    match_start = min(match_start, text_length);
    match_end = min(match_end, text_length);

    // Initialize shared memory.
    for (int i = threadIdx.x; 4*i < pattern_length; i += blockDim.x) {
        *(int *)(shared_pattern + 4*i) = *(int *)(pattern + 4*i);
    }
    for (int i = threadIdx.x; i <= pattern_length + 1; i += blockDim.x) {
        shared_fail[i] = fail[i];
    }
    if (threadIdx.x == 0) {
        *shared_output_cnt = 0;
    }
    __syncthreads();

    KMP_search(
        text, match_start, match_end, shared_pattern, pattern_length,
        shared_output, shared_output_cnt, output, output_cnt, max_output_cnt,
        shared_fail
    );

    // Write the outputs to the global memory.
    int block_output_cnt = min(*shared_output_cnt, output_cache_size_per_block);
    if (threadIdx.x == 0) {
        *shared_global_output_start = atomicAdd(output_cnt, block_output_cnt);
    }
    __syncthreads();

    int global_output_start = *shared_global_output_start;
    for (
        int i = threadIdx.x;
        i < block_output_cnt && i + global_output_start < max_output_cnt;
        i += blockDim.x
    ) {
        output[i + global_output_start] = shared_output[i];
    }
}

int KMP_search_shmem(
    const char *text, int text_length, const char *pattern, int pattern_length,
    int *output, int max_output_cnt, int *fail
) {
    if (match_length_per_thread <= pattern_length) {
        LOG(error, "match_length_per_thread should be larger than pattern_length");
        exit(1);
    }

    // Array sizes, in bytes.
    int text_size = round_up_to_multiple(ceil_div(text_length * sizeof(char), 4), sizeof(int));
    int pattern_size = round_up_to_multiple(ceil_div(pattern_length * sizeof(char), 4), sizeof(int));
    int fail_size = sizeof(int) * (pattern_length + 1);
    int output_size = sizeof(int) * max_output_cnt;

    timer_start("Allocating GPU memory");
    char *text_device;
    char *pattern_device;
    int *output_device;
    int *fail_device;
    int *output_cnt_device;
    THROW_IF_ERROR(cudaMalloc((void **)&text_device, text_size));
    THROW_IF_ERROR(cudaMalloc((void **)&pattern_device, pattern_size));
    THROW_IF_ERROR(cudaMalloc((void **)&output_device, output_size));
    THROW_IF_ERROR(cudaMalloc((void **)&fail_device, fail_size));
    THROW_IF_ERROR(cudaMalloc((void **)&output_cnt_device, sizeof(int)));
    timer_stop();

    timer_start("Computing fail array on the CPU");
    get_fail(pattern, pattern_length, fail);
    timer_stop();

    timer_start("Copying inputs to the GPU");
    THROW_IF_ERROR(cudaMemcpy(text_device, text, text_size, cudaMemcpyHostToDevice));
    THROW_IF_ERROR(cudaMemcpy(pattern_device, pattern, pattern_size, cudaMemcpyHostToDevice));
    THROW_IF_ERROR(cudaMemcpy(fail_device, fail, fail_size, cudaMemcpyHostToDevice));
    THROW_IF_ERROR(cudaMemset(output_cnt_device, 0, sizeof(int)));
    timer_stop();

    // Prepare to launch the kernel.
    int shared_memory_size =
        pattern_size + fail_size
        + sizeof(int) * output_cache_size_per_block  // per block output cache.
        + sizeof(int)  // per block output counter.
        + sizeof(int);  // per block output start index.

    int num_blocks = ceil_div(
        text_length - (pattern_length - 1),
        block_size * (match_length_per_thread - (pattern_length - 1))
    );
    printf("pattern size %d, fail size %d\n", pattern_size, fail_size);
    printf("shared memory size %d\n", shared_memory_size);

    timer_start("Performing KMP on the GPU");
    KMP_search_shmem_kernel<<<num_blocks, block_size, shared_memory_size>>>(
        text_device, text_length, pattern_device, pattern_length,
        output_device, output_cnt_device, max_output_cnt, fail_device
    );
    THROW_IF_ERROR(cudaDeviceSynchronize());
    timer_stop();

    timer_start("Copying output memory to the CPU");
    int output_cnt;
    THROW_IF_ERROR(
        cudaMemcpy(&output_cnt, output_cnt_device, sizeof(int), cudaMemcpyDeviceToHost)
    );
    output_cnt = min(output_cnt, max_output_cnt);  // This counter may overflow.
    THROW_IF_ERROR(cudaMemcpy(output, output_device, output_size, cudaMemcpyDeviceToHost));
    timer_stop();

    cudaFree(text_device);
    cudaFree(pattern_device);
    cudaFree(output_device);
    cudaFree(fail_device);
    cudaFree(output_cnt_device);
    return output_cnt;
}
