#include "state_machine_shmem.cuh"

#include <vector>
#include "common/utils.hpp"
#include "kmp_cpu.hpp"
#include "state_machine_cpu.hpp"

// This value should be several times larger than pattern_length.
static constexpr int match_length_per_thread = 128;

static constexpr int block_size = 128;

static __host__ __device__ int ceil_div(int x, int y) {
    return (x - 1) / y + 1;
}

// Get arr[idx], where arr is the compact form of the gene sequence.
template<class IdxType>
static __host__ __device__ inline char get(const char *arr, IdxType idx) {
    return (arr[idx>>2] >> ((idx & 0x3) << 1)) & 0x3;
}

__device__ static void state_machine_search(
    const char *text, const int text_length, int match_start, int match_end,
    const int16_t pattern_length, int *output, int *output_cnt, const int max_output_cnt,
    const int16_t (*jump_table)[4]
) {
    if (match_start < match_end) {
        int i = match_start;
        int curr_state = 0;
        char packed_text = text[i >> 2] >> ((i & 0x3) << 1);
        for (; i < match_end; i++, packed_text >>=2) {
            if (!(i & 0x3)) {
                packed_text = text[i >> 2];
            }
            curr_state = jump_table[curr_state][packed_text & 0x3];
            if (curr_state == pattern_length) {
                int outputIdx = atomicAdd(output_cnt, 1);
                if (outputIdx >= max_output_cnt) {
                    return;
                }
                output[outputIdx] = i - pattern_length + 1;
            }
        }
    }
}

__global__ void state_machine_search_shmem_kernel(
    const char *text, const int text_length, const int16_t pattern_length,
    int *output, int *output_cnt, const int max_output_cnt, int16_t (*jump_table)[4]
) {
    extern __shared__ int16_t shared_jump_table[][4];

    int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    int match_start = (match_length_per_thread - (pattern_length - 1)) * global_index;
    int match_end = match_start + match_length_per_thread;
    match_start = min(match_start, text_length);
    match_end = min(match_end, text_length);

    // Initialize shared memory.
    for (int i = threadIdx.x; i < 4 * (pattern_length + 1); i += blockDim.x) {
        shared_jump_table[i>>2][i&0x3] = jump_table[i>>2][i&0x3];
    }
    __syncthreads();

    state_machine_search(
        text, text_length, match_start, match_end,
        pattern_length, output, output_cnt, max_output_cnt, shared_jump_table
    );
}

int state_machine_search_shmem(
    const char *text, int text_length, const char *pattern, int16_t pattern_length,
    int *output, int max_output_cnt, int16_t *fail
) {
    if (match_length_per_thread <= pattern_length) {
        fprintf(stderr, "match_length_per_thread should be larger than pattern_length");
        exit(1);
    }

    timer_start("Computing state machine jump table on the CPU");
    std::vector<int16_t> jump_table(4 * (pattern_length+1));
    get_fail(pattern, pattern_length, fail);
    build_state_machine(
        reinterpret_cast<int16_t (*)[4]>(jump_table.data()),
        pattern, fail, pattern_length
    );
    timer_stop();

    // Array sizes, in bytes.
    int text_size = ceil_div(text_length * sizeof(char), 4);
    int jump_table_size = sizeof(int16_t) * (pattern_length + 1) * 4;
    int output_size = sizeof(int) * max_output_cnt;

    timer_start("Allocating GPU memory");
    char *text_device;
    int *output_device;
    int16_t *jump_table_device;
    int *output_cnt_device;
    THROW_IF_ERROR(cudaMalloc((void **)&text_device, text_size));
    THROW_IF_ERROR(cudaMalloc((void **)&output_device, output_size));
    THROW_IF_ERROR(cudaMalloc((void **)&jump_table_device, jump_table_size));
    THROW_IF_ERROR(cudaMalloc((void **)&output_cnt_device, sizeof(int)));
    timer_stop();

    timer_start("Copying inputs to the GPU");
    THROW_IF_ERROR(cudaMemcpy(text_device, text, text_size, cudaMemcpyHostToDevice));
    THROW_IF_ERROR(cudaMemcpy(jump_table_device, jump_table.data(), jump_table_size, cudaMemcpyHostToDevice));
    THROW_IF_ERROR(cudaMemset(output_cnt_device, 0, sizeof(int)));
    timer_stop();

    // Prepare to launch the kernel.

    int num_blocks = ceil_div(
        text_length - (pattern_length - 1),
        block_size * (match_length_per_thread - (pattern_length - 1))
    );

    timer_start("Performing state machine search on the GPU");
    state_machine_search_shmem_kernel<<<num_blocks, block_size, jump_table_size>>>(
        text_device, text_length, pattern_length,
        output_device, output_cnt_device, max_output_cnt, reinterpret_cast<int16_t (*)[4]>(jump_table_device)
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

    timer_start("Freeing GPU memory");
    cudaFree(text_device);
    cudaFree(output_device);
    cudaFree(jump_table_device);
    cudaFree(output_cnt_device);
    timer_stop();
    return output_cnt;
}
