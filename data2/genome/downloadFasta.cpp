#include <iostream>
#include <fstream>
#include <string>
#include <map>

void downloadDataset()
{

    // cmd: wget -O genome.fasta https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_000001.11&rettype=fasta&retmode=text
    const std::string url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_000001.11&rettype=fasta&retmode=text";
    const std::string output_filename = "genome.fasta";

    std::string command = "wget -O " + output_filename + " " + url;
    std::cout << "Running command:" << command << std::endl;

    int result = system(command.c_str());

    if (result != 0)
    {
        std::cerr << "Error executing wget command. Return code: " << result << std::endl;
    }
    else
    {
        std::cout << "File downloaded successfully: " << output_filename << std::endl;
    }
}

int main()
{
    downloadDataset();
    return 0;
}