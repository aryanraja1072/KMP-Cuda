#pragma once

int state_machine_search_async_copy(
    const char *text, int text_length, const char *pattern, int pattern_length,
    int *output, int max_output_cnt, int *fail
);
