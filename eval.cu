#include <cstring>
#include <string>
#include <random>
#include <vector>

#include "kmp_cpu.hpp"

void generate_data(char *data, int length, const char *alphabet) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, strlen(alphabet)-1);
    for (int i = 0; i < length; i++) {
        data[i] = alphabet[distrib(gen)];
    }
}

void eval_with_random_input(
    int text_length, int pattern_length, const char *alphabet="ATCG"
) {
    std::string text(text_length, '\0');
    std::string pattern(pattern_length, '\0');
    std::vector<int> output(100);
    std::vector<int> fail(pattern_length + 1);

    generate_data(&text[0], text_length, alphabet);
    generate_data(&pattern[0], pattern_length, alphabet);

    int cnt = KMP_search(
        text.data(), text.length(), pattern.data(), pattern.length(),
        output.data(), output.size(), fail.data()
    );
    printf("%d matches found\n", cnt);
}

int main() {
    eval_with_random_input(8000000, 10);
    return 0;
}
