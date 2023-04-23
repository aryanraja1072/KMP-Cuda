#pragma once

#include <cstdint>

// Find at most `max_output_count` occurrences of pattern in text, and store
// them in the array `output`.
// This brute force solution is only used to check correctness.
int brute_force_search(
    const char *text, int text_length, const char *pattern, int16_t pattern_length,
    int16_t *output, int max_output_cnt, int16_t *fail
);
