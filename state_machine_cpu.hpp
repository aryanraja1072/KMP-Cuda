#pragma once

#include <cstdint>

void build_state_machine(
    int16_t (*state_machine)[4],
    const char *pattern, const int16_t *fail, int16_t pattern_length
);

int state_machine_search(
    const char *text, int text_length, const char *pattern, int16_t pattern_length,
    int *output, int max_output_cnt, int16_t *fail
);
