#include <iostream>
#include <fstream>
#include <string>
#include <map>

/*
Source: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_000001.11&rettype=fasta&retmode=text (NCBI)

The content of a FASTA file is organized as follows:

Sequence identifier: Each sequence in the file begins with a single-line description, called the sequence identifier. The identifier line starts with the '>' character, followed by a unique identifier and, optionally, a description of the sequence. The identifier can include alphanumeric characters, underscores, and periods. The description can contain any additional information about the sequence, such as its source or function.

Sequence data: Following the sequence identifier line is the actual sequence data, represented as a series of characters. In the case of DNA sequences, the characters represent the four nucleotides: adenine (A), cytosine (C), guanine (G), and thymine (T). The sequence data can span multiple lines, with each line usually having a fixed number of characters (e.g., 60 or 80 characters per line). Lines should not contain any whitespace or special characters other than the sequence characters.

A stretch of 'N's in a FASTA file or genomic data means that the specific nucleotide sequence is not known or could not be accurately determined during the sequencing process.

There are several reasons why 'N's might appear in a DNA sequence:

Sequencing errors or limitations: Some DNA sequencing techniques, such as Sanger sequencing or Illumina sequencing, might produce errors or have limitations in resolving certain regions of the genome. In these cases, the unknown nucleotides are represented as 'N's.

Low sequence coverage or quality: If the sequencing coverage is low or the quality of the sequence data is poor in a particular region, it might be difficult to determine the exact nucleotide composition. As a result, ambiguous nucleotides are represented as 'N's.

Repetitive regions or gaps in the assembly: Some regions of the genome contain highly repetitive sequences, which can be challenging to assemble accurately. In these cases, the repetitive regions or gaps in the assembly may be represented by 'N's.

Masking specific sequences: Sometimes, researchers might mask certain sequences in the genome data, such as repetitive elements or regions with potential privacy concerns, by replacing the original nucleotides with 'N's.
*/

/*
 Reads a FASTA file line by line. When a line starts with the > character,
 it's considered a sequence identifier line, and
 the function extracts the identifier and initializes an empty sequence for that identifier in the sequences map.
 For other lines, it appends the DNA sequence to the current identifier's sequence in the map.
 The function returns a map containing all the sequences from the FASTA file.
*/

std::map<std::string, std::string> readFastaFile(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return {};
    }

    std::map<std::string, std::string> sequences;
    std::string line;
    std::string current_id;
    while (std::getline(file, line))
    {
        if (line.empty())
        {
            continue;
        }

        if (line[0] == '>')
        { // Sequence identifier line
            current_id = line.substr(1);
            sequences[current_id] = "";
        }
        else
        { // DNA sequence line
            sequences[current_id] += line;
        }
    }

    file.close();
    return sequences;
}

int main()
{
    const std::string fasta_filename = "genome.fasta";
    std::map<std::string, std::string> sequences = readFastaFile(fasta_filename);

    for (const auto &sequence : sequences)
    {
        std::cout << "Sequence ID: " << sequence.first << std::endl;
        std::cout << "Sequence: " << sequence.second << std::endl
                  << std::endl;
    }

    return 0;
}
