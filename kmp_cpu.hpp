#pragma once

// Find at most `max_output_count` occurrences of pattern in text, and store
// them in the array `output`.
// The `fail` array is pre-allocated, with length `pattern_length+1`.
// Return the number of occurrences of the pattern.
int KMP_search(
    const char *text, int text_length, const char *pattern, int pattern_length,
    int *output, int max_output_cnt, int *fail
);
