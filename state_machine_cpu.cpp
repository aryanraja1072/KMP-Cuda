#include "state_machine_cpu.hpp"
#include "kmp_cpu.hpp"
#include "char_compress.hpp"

void build_state_machine(
    int16_t (*state_machine)[4],
    const char *pattern, const int16_t *fail, int16_t pattern_length
) {
    for (int16_t i = 0; i <= pattern_length; i++) {
        for (char j = 0; j < 4; j++) {
            if (i < pattern_length && get(pattern, i) == j) {
                // Matches, go to the next state.
                state_machine[i][j] = i + 1;
            }
            else {
                int k = fail[i];
                while (k >= 0 && get(pattern, k) != j) {
                    k = fail[k];
                }
                state_machine[i][j] = k + 1;
            }
        }
    }
}

int state_machine_search(
    const char *text, int text_length, const char *pattern, int16_t pattern_length,
    int *output, int max_output_cnt, int16_t *fail
) {
    get_fail(pattern, pattern_length, fail);

    auto state_machine = new int16_t[pattern_length+1][4];
    build_state_machine(state_machine, pattern, fail, pattern_length);

    int16_t curr_state = 0;
    int output_cnt = 0;
    for (int i = 0; i < text_length; i += 4) {
        char text_packed = text[i >> 2];
        for (int ii = 0; ii < 4 && i + ii < text_length; ii++, text_packed >>= 2) {
            curr_state = state_machine[curr_state][text_packed & 0x3];
            if (curr_state == pattern_length) {
                output[output_cnt++] = i + ii - pattern_length;
                if (output_cnt == max_output_cnt) {
                    goto end;  // A simple way to break double loop.
                }
            }
        }
    }

end:
    delete[] state_machine;
    return output_cnt;
}
