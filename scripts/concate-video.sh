#!/bin/bash

# Define the output directory and output video file name
output_directory="inference_results"  # Change this to your actual output directory
output_file="output_video.mp4"

# Create a file list for ffmpeg
file_list="file_list.txt"

# Remove the previous file list if it exists
rm -f $file_list

# Generate the file list
for i in {0..26}; do
    echo "file '$output_directory/frame_result_$i.mp4'" >> $file_list
done

# Concatenate the videos using ffmpeg
ffmpeg -f concat -safe 0 -i $file_list -c copy "$output_directory/$output_file"

# Clean up the file list
rm -f $file_list

echo "Concatenation complete! Output video saved as $output_directory/$output_file"