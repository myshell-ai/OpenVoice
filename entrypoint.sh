#!/bin/sh

download_and_unzip() {
    url=$1
    output_zip=$2
    output_dir=$3
    
    wget $url -O $output_zip && \
    mkdir -p /tmp/extract_temp && \
    unzip $output_zip -d /tmp/extract_temp && \
    first_folder=$(find /tmp/extract_temp -mindepth 1 -maxdepth 1 -type d | head -n 1) && \
    mkdir -p $output_dir && \
    mv $first_folder/* $output_dir/ && \
    rm -rf /tmp/extract_temp && \
    rm $output_zip
}

if [ "$1" = "v1" ]; then
    echo "Downloading v1 checkpoints..."
    download_and_unzip "https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/checkpoints_1226.zip" "checkpoints_1226.zip" "/workspace/checkpoints_v1"
elif [ "$1" = "v2" ]; then
    echo "Downloading v2 checkpoints..."
    download_and_unzip "https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip" "checkpoints_v2_0417.zip" "/workspace/checkpoints_v2"
else
    echo "Downloading both v1 and v2 checkpoints..."
    download_and_unzip "https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/checkpoints_1226.zip" "checkpoints_1226.zip" "/workspace/checkpoints_v1"
    download_and_unzip "https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip" "checkpoints_v2_0417.zip" "/workspace/checkpoints_v2"
fi
echo "Starting Jupyter Notebook..."
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.notebook_dir='/workspace' &

wait
