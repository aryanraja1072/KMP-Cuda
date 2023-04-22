#include "brute_force.hpp"

// Get arr[idx], where arr is the compact form of the gene sequence.
static inline char get(const char *arr, int idx) {
    return (arr[idx>>2] >> ((idx & 0x3) << 1)) & 0x3;
}

int brute_force_search(
    const char *text, int text_length, const char *pattern, int pattern_length,
    int *output, int max_output_cnt, int *fail
) {
    int output_cnt = 0;
    for (int i = 0; i < text_length - pattern_length; i++) {
        bool match = true;
        for (int j = 0; j < pattern_length; j++) {
            if (get(text, i+j) != get(pattern, j)) {
                match = false;
                break;
            }
        }
        if (match) {
            if (output_cnt >= max_output_cnt) {
                break;
            }
            output[output_cnt++] = i;
        }
    }
    return output_cnt;
}
