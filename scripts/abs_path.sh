#!/bin/bash

# Check if at least one argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 relative/path"
    exit 1
fi

# Convert the relative path to an absolute path
absolute_path=$(realpath "$1")

# Print the absolute path
echo "Absolute path: $absolute_path"
