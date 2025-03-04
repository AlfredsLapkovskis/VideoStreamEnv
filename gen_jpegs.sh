#!/bin/bash

# Check if the video name is provided as a parameter
if [ -z "$1" ]; then
  echo "Usage: $0 <video_file>"
  exit 1
fi

# Assign the input video file from the first argument
input="$1"

# Verify if the input file exists
if [ ! -f "$input" ]; then
  echo "Error: File '$input' not found!"
  exit 1
fi

# Extract the base name of the video file (without the extension)
base_name=$(basename "$input" | sed 's/\.[^.]*$//')

# Loop through the desired resolutions
for size in 180 360 720; do
  # Create the directory structure
  output_dir="videos/${base_name}/${size}x${size}"
  mkdir -p "$output_dir"
  
  # Generate JPEGs in the directory
  ffmpeg -i "$input" -vf "scale=${size}:${size}:force_original_aspect_ratio=increase,crop=${size}:${size}" -q:v 2 "${output_dir}/%04d.jpg"
done
