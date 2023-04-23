#!/bin/bash

echo "Dowloading Chromosome 1"
wget "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_000001.11&rettype=fasta&retmode=text" -O chromosome_1.fasta
echo "Dowloading Chromosome 2"
wget "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_000002.12&rettype=fasta&retmode=text" -O chromosome_2.fasta
echo "Downloading Chromosome 3"
wget "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_000003.12&rettype=fasta&retmode=text" -O chromosome_3.fasta
echo "Downloading Chromosome 4"
wget "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_000004.12&rettype=fasta&retmode=text" -O chromosome_4.fasta
echo "Downloading Chromosome 5"
wget "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_000005.10&rettype=fasta&retmode=text" -O chromosome_5.fasta
echo "Downloading Chromosome 6"
wget "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_000006.12&rettype=fasta&retmode=text" -O chromosome_6.fasta






