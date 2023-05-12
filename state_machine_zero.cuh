#pragma once

int state_machine_search_zerocopy(
    const char *text, int text_length, const char *pattern, int16_t pattern_length,
    int *output, int max_output_cnt, int16_t *fail);
