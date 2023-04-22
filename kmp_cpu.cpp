#include "kmp_cpu.hpp"

static void get_fail(const char *pattern, int pattern_length, int *fail) {
    int candidate = 0;
    fail[0] = -1;
    unsigned int clear_mask = 0x00000011;
    for (int pos = 1; pos < pattern_length; pos++, candidate++) {
        unsigned char com_pos = (pattern[pos / 4] >> (2 * (pos % 4))) |= clear_mask;
        unsigned char com_cand = (pattern[candidate / 4] >> (2 * (candidate % 4))) |= clear_mask;
        if (com_pos == com_cand) {
            fail[pos] = fail[candidate];
        }
        else {
            fail[pos] = candidate;
            while (candidate >= 0 && com_pos != com_cand) {
                candidate = fail[candidate];
            }
        }
    }
    fail[pattern_length] = candidate;
}

void char_compress(const char *text, int text_length, char& text_com) {
    // A: 00; B: 01; C:10; D: 11 
    unsigned int A_mask = 0x00000000;
    unsigned int T_mask = 0x00000001;
    unsigned int C_mask = 0x00000010;
    unsigned int G_mask = 0x00000011;
    for (int idx = 0; idx < text_length; idx++) {
        int i = idx / 4;
        int j = idx % 4;
        char text_tmp = text[4 * i + j];
        if (text_tmp == 'A') {
            text_com[i] |= (A_mask << (2 * j));
        }
        else if (text_tmp == 'T') {
            text_com[i] |= (B_mask << (2 * j));
        }
        else if (text_tmp == 'C') {
            text_com[i] |= (C_mask << (2 * j));
        }
        else if (text_tmp == 'G') {
            text_com[i] |= (D_mask << (2 * j));
        }
    }
}

int KMP_search(
    const char *text, int16 text_length, const char *pattern, int16 pattern_length,
    int16 *output, int16 max_output_cnt, int16 *fail
) {
    char * text_com = (char*)calloc((text_length / 4 + 1), sizeof(char));
    char_compress(text, text_length, text_com)
    char * pattern_com = (char*)calloc((pattern_length / 4 + 1), sizeof(char));
    char_compress(pattern, pattern_length, pattern_com)

    get_fail(pattern_com, pattern_length, fail);

    int i = 0, j = 0;
    int output_cnt = 0;
    unsigned int clear_mask = 0x00000011
    while (i < text_length) {
        unsigned char com_text = (text_com[i / 4] >> (2 * (i % 4))) |= clear_mask
        unsigned char com_pat = (pattern_com[i / 4] >> (2 * (i % 4))) |= clear_mask
        if (com_text == com_pat) {
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
