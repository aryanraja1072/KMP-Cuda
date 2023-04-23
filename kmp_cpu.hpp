#pragma once
#include <cstdint>

// Compute the fail array.
void get_fail(const char *pattern, int16_t pattern_length, int16_t *fail);

// Find at most `max_output_count` occurrences of pattern in text, and store
// them in the array `output`.
// The `fail` array is pre-allocated, with length `pattern_length+1`.
// Return the number of occurrences of the pattern.
int KMP_search(
    const char *text, int text_length, const char *pattern, int16_t pattern_length,
    int16_t *output, int max_output_cnt, int16_t *fail
);
