#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>

std::string read_fasta_data(const std::string &filename)
{
    std::ifstream file(filename, std::ios::in);
    std::string data;

    if (!file)
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return data;
    }

    std::string line;
    while (std::getline(file, line))
    {
        if (line.empty() || line[0] == '>')
        { // empty line or sequence identifier line
            continue;
        }

        data += line;
    }

    file.close();
    return data;
}

std::string load_data(size_t length)
{
    std::string data;
    data.reserve(length);

    for (const auto &entry : std::filesystem::directory_iterator("."))
    {
        if (entry.path().extension() == ".fasta")
        {
            std::string file_data = read_fasta_data(entry.path().string());

            size_t remaining_length = length - data.size();
            data.append(file_data, 0, remaining_length);

            if (data.size() >= length)
            {
                break;
            }
        }
    }

    return data;
}

void write_data_to_file(const std::string &output_filename, const std::string &data)
{
    std::ofstream output_file(output_filename, std::ios::out);

    if (!output_file)
    {
        std::cerr << "Error opening output file: " << output_filename << std::endl;
        return;
    }

    output_file << data;
    output_file.close();
}

int main(int argc, char *argv[])
{
    size_t length = 100;
    std::string output_filename = "data.txt";
    if (argc >= 2)
    {

        try
        {
            length = std::stoul(argv[1]);
            std::cout << "Length: " << length << std::endl;
        }
        catch (const std::invalid_argument &e)
        {
            std::cerr << "Error: Invalid argument. Please provide an unsigned integer value." << std::endl;
            return 1;
        }
        catch (const std::out_of_range &e)
        {
            std::cerr << "Error: Argument out of range for size_t." << std::endl;
            return 1;
        }
    }

    if (argc == 3)
    {
        output_filename = argv[2];
        std::cout << "Output filename: " << output_filename << std::endl;
    }

    std::string data = load_data(length);
    write_data_to_file(output_filename, data);

    std::cout << "Data saved to: " << output_filename << std::endl;

    return 0;
}
