#include "kmp_cpu.hpp"
#include "char_compress.hpp"

void get_fail(const char *pattern, int16_t pattern_length, int16_t *fail) {
    int candidate = 0;
    fail[0] = -1;
    for (int pos = 1; pos < pattern_length; pos++, candidate++) {
        char com_pos = get(pattern, pos);
        char com_cand = get(pattern, candidate);
        if (com_pos == com_cand) {
            fail[pos] = fail[candidate];
        }
        else {
            fail[pos] = candidate;
            while (candidate >= 0 && com_pos != get(pattern, candidate)) {
                candidate = fail[candidate];
            }
        }
    }
    fail[pattern_length] = candidate;
}

int KMP_search(
    const char *text, int text_length, const char *pattern, int16_t pattern_length,
    int16_t *output, int max_output_cnt, int16_t *fail
) {
    get_fail(pattern, pattern_length, fail);
    int i = 0, j = 0;
    int output_cnt = 0;
    while (i < text_length) {
        if (get(text, i) == get(pattern, j)) {
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
