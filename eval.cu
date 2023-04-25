#include <algorithm>
#include <cstring>
#include <chrono>
#include <string>
#include <random>
#include <vector>

#include "char_compress.hpp"
#include "brute_force.hpp"
#include "kmp_cpu.hpp"
#include "kmp_shmem.cuh"

using StringMatchingFunction = decltype(brute_force_search);

void generate_random_data(char *data, int length) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<char> distrib(-128, 127);
    for (int i = 0; 4*i < length; i++) {
        data[i] = distrib(gen);
    }
}

void print_text_or_pattern(const std::string &text_or_pattern, int length) {
    for (int i = 0; i < length; i++) {
        char char_compressed = get(text_or_pattern.data(), i);
        if (char_compressed == static_cast<char>(Gene::A)) {
            printf("A");
        }
        else if (char_compressed == static_cast<char>(Gene::T)) {
            printf("T");
        }
        else if (char_compressed == static_cast<char>(Gene::C)) {
            printf("C");
        }
        else if (char_compressed == static_cast<char>(Gene::G)) {
            printf("G");
        }
    }
    printf(" (hex = ");
    for (int i = 0; 4*i < length; i++) {
        printf("%02hhx", text_or_pattern[i]);
    }
    printf(")");
}

void print_output(std::vector<int> &output, int output_cnt) {
    printf("%d matches found\n", output_cnt);
    if (output_cnt > 0) {
        printf("matching at: ");
        for (int i = 0; i < output_cnt; i++) {
            printf("%d ", output[i]);
        }
        printf("\n");
    }
}

// Returns 0 for success, -1 for failure.
int compare_result(
    const std::vector<int> &output, int cnt,
    const std::vector<int> &ref_output, int ref_cnt,
    int max_output_cnt
) {
    if (cnt != ref_cnt) {
        printf("Pattern count mismatch! %d (tested) != %d (ref)!\n", cnt, ref_cnt);
        return -1;
    }
    if (cnt == max_output_cnt) {
        printf(
            "Warning: output limit is reached! The GPU and CPU algorithms may produce different result.\n"
            "Results will not be compared this time. You should set a larger max_output_cnt and try again.\n"
        );
        return 0;
    }

    std::vector<int> sorted_output = output;
    std::sort(sorted_output.begin(), sorted_output.begin() + cnt);
    std::vector<int> sorted_ref_output = ref_output;
    std::sort(sorted_ref_output.begin(), sorted_ref_output.begin() + ref_cnt);
    for (int i = 0; i < cnt; i++) {
        if (sorted_output[i] != sorted_ref_output[i]) {
            printf("Mismatch at the %d-th occurrence, %d (tested) != %d (ref)!\n", i, sorted_output[i], sorted_ref_output[i]);
            return -1;
        }
    }
    printf("Results are the same!\n");
    return 0;
}

// Returns 0 for success, -1 for failure.
int eval(
    const std::string &text, int text_length,
    std::string &pattern, int16_t pattern_length,
    StringMatchingFunction search_function,
    StringMatchingFunction reference_function=nullptr,
    bool verbose=false, int max_output_cnt=100
) {
    if (verbose) {
        printf("text = ");
        print_text_or_pattern(text, text_length);
        printf("\n");

        printf("pattern = ");
        print_text_or_pattern(pattern, pattern_length);
        printf("\n");
    }

    std::vector<int> output(max_output_cnt);
    std::vector<int16_t> fail(pattern_length + 1);

    auto start = std::chrono::steady_clock::now();
    int cnt = search_function(
        text.data(), text_length, pattern.data(), pattern_length,
        output.data(), output.size(), fail.data()
    );
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> diff = end - start;
    printf("Search function takes %lf ms.\n", diff.count());

    if (verbose) {
        printf("test function output: ");
        print_output(output, cnt);
    }

    if (reference_function != nullptr) {
        std::vector<int> ref_output(max_output_cnt);

        auto start = std::chrono::steady_clock::now();
        int ref_cnt = reference_function(
            text.data(), text_length, pattern.data(), pattern_length,
            ref_output.data(), ref_output.size(), fail.data()
        );
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> diff = end - start;
        printf("Reference function takes %lf ms.\n", diff.count());

        if (verbose) {
            printf("reference function output: ");
            print_output(ref_output, ref_cnt);
        }

        return compare_result(output, cnt, ref_output, ref_cnt, max_output_cnt);
    }
    return 0;
}

// Returns 0 for success, -1 for failure.
int eval_with_random_input(
    int text_length, int pattern_length,
    StringMatchingFunction search_function,
    StringMatchingFunction reference_function=nullptr,
    bool verbose=false, int max_output_cnt=100
) {
    std::string text((text_length-1)/4+1, '\0');
    std::string pattern((pattern_length-1)/4+1, '\0');

    generate_random_data(&text[0], text_length);
    generate_random_data(&pattern[0], pattern_length);

    return eval(
        text, text_length, pattern, pattern_length,
        search_function, reference_function, verbose, max_output_cnt
    );
}

// Returns 0 for success, -1 for failure.
int eval_with_string_data(
    const char *text, const char *pattern,
    StringMatchingFunction search_function,
    StringMatchingFunction reference_function=nullptr,
    bool verbose=false, int max_output_cnt=100
) {
    int text_length = strlen(text);
    int pattern_length = strlen(pattern);
    std::string compressed_text((text_length-1)/4+1, '\0');
    std::string compressed_pattern((pattern_length-1)/4+1, '\0');

    char_compress(text, text_length, &compressed_text[0]);
    char_compress(pattern, pattern_length, &compressed_pattern[0]);

    return eval(
        compressed_text, text_length,
        compressed_pattern, pattern_length,
        search_function, reference_function, verbose, max_output_cnt
    );
}

int main() {
    for (int i = 0; i < 6; i++) {
        eval_with_random_input(
            500000000, 12, KMP_search_shmem, nullptr, /*verbose=*/false
        );
    }
}
