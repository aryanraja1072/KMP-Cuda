#!/bin/bash

# source : https://drive.google.com/file/d/1-fg41lJZlGNOPZhR9uP3qviIblMhhX6k/view?usp=sharing 


file_id="1-fg41lJZlGNOPZhR9uP3qviIblMhhX6k"
output_file="hdfs_logs.tar.gz"

# Function to extract the Google Drive API response's confirm code
extract_confirm_code() {
    grep -o 'confirm=[^&]*' | sed 's/confirm=//'
}

if command -v curl >/dev/null 2>&1; then
    # Get the confirm code
    confirm_code=$(curl -sc /tmp/gcookie "https://drive.google.com/uc?export=download&id=${file_id}" | extract_confirm_code)
    # Download the file using the confirm code
    curl -Lb /tmp/gcookie "https://drive.google.com/uc?export=download&confirm=${confirm_code}&id=${file_id}" -o "${output_file}"
elif command -v wget >/dev/null 2>&1; then
    # Get the confirm code
    confirm_code=$(wget --quiet --save-cookies /tmp/gcookie --keep-session-cookies --no-check-certificate "https://drive.google.com/uc?export=download&id=${file_id}" -O- | extract_confirm_code)
    # Download the file using the confirm code
    wget --quiet --load-cookies /tmp/gcookie "https://drive.google.com/uc?export=download&confirm=${confirm_code}&id=${file_id}" -O "${output_file}"
else
    echo "Error: Neither 'curl' nor 'wget' found. Please install one of them and try again."
    exit 1
fi

echo "File downloaded as: ${output_file}"
