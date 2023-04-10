#include "kmp_cpu.hpp"

static void get_fail(const char *pattern, int pattern_length, int *fail) {
    int candidate = 0;
    fail[0] = -1;
    for (int pos = 1; pos < pattern_length; pos++, candidate++) {
        if (pattern[pos] == pattern[candidate]) {
            fail[pos] = fail[candidate];
        }
        else {
            fail[pos] = candidate;
            while (candidate >= 0 && pattern[pos] != pattern[candidate]) {
                candidate = fail[candidate];
            }
        }
    }
    fail[pattern_length] = candidate;
}

int KMP_search(
    const char *text, int text_length, const char *pattern, int pattern_length,
    int *output, int max_output_cnt, int *fail
) {
    get_fail(pattern, pattern_length, fail);

    int i = 0, j = 0;
    int output_cnt = 0;
    while (i < text_length) {
        if (text[i] == pattern[j]) {
            i++; j++;
            if (j == pattern_length) {
                // Occurrence found.
                output[output_cnt++] = i - j;
                if (output_cnt == max_output_cnt) {
                    break;
                }
            }
        }
        else {
            j = fail[j];
            if (j < 0) {
                i++, j++;
            }
        }
    }
    return output_cnt;
}
