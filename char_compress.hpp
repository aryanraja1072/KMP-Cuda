#pragma once
void char_compress(const char *text, int text_length, char * text_com);

enum class Gene : char {
     A = 0b00000000,
     T = 0b00000001,
     C = 0b00000010,
     G = 0b00000011,
};

char get(const char *arr, int idx);
