#!/bin/bash


input_folder="Performence_Analysis/Main_input"


output_folder="outputfolder-6"


mkdir -p "$output_folder"


for input_file in "$input_folder"/*; do
 
    filename=$(basename -- "$input_file")
    filename_no_ext="${filename%.*}"

    output_file="$output_folder/${filename_no_ext}_out.jpg"

    
    python main.py "$input_file" "$output_file"

    echo "Processed: $input_file"
done
