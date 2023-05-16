#pragma once

// The baseline KMP search algorithm as stated in the paper. Shared memory is
// used to store `pattern` and `fail`.
int KMP_search_shmem(
    const char *text, int text_length, const char *pattern, int pattern_length,
    int *output, int max_output_cnt, int *fail
);
