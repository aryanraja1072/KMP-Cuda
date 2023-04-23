#include "char_compress.hpp"

void char_compress(const char *text, int text_length, char * text_com) {
    for (int idx = 0; idx < text_length; idx++) {
        int i = idx / 4;
        int j = idx % 4;
        char text_tmp = text[idx];
        if (text_tmp == 'A') {
            text_com[i] = text_com[i] | (Gene::A << (2 * j));
        }
        else if (text_tmp == 'T') {
            text_com[i] = text_com[i] | (Gene::T << (2 * j));
        }
        else if (text_tmp == 'C') {
            text_com[i] = text_com[i] | (Gene::C << (2 * j));
        }
        else if (text_tmp == 'G') {
            text_com[i] = text_com[i] | (Gene::G << (2 * j));
        }
    }

}

// Get arr[idx], where arr is the compact form of the gene sequence.
inline char get(const char *arr, int idx) {
    return (arr[idx>>2] >> ((idx & 0x3) << 1)) & 0x3;
}
