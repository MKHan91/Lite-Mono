#!/bin/bash

input_file="download_train.txt"
base_url="https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"
download_dir="/home/dev/DATASET/kitti_data"

while read -r line; do
    # Split by space
    path=$(echo "$line" | cut -d' ' -f1)
    
    # Extract only the folder name, which is the zip name
    zip_name=$(basename "$path")

    folder_name=${zip_name%_sync}

    # Build URL
    url="$base_url/$folder_name/$zip_name.zip"

    echo "Downloading: $url"
    wget -P "$download_dir" "$url"
    
done < "$input_file"
