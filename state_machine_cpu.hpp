#pragma once

#include <cstdint>

void build_state_machine(
    int (*state_machine)[4],
    const char *pattern, const int *fail, int pattern_length
);

int state_machine_search(
    const char *text, int text_length, const char *pattern, int pattern_length,
    int *output, int max_output_cnt, int *fail
);
