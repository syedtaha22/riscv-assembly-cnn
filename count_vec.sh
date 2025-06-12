#!/bin/bash

# Help message function
show_help() {
    echo "Usage: $0 <directory>"
    echo
    echo "Counts vector instructions (e.g., vadd, vle, etc.) in all .s files within the given directory."
    echo
    echo "Arguments:"
    echo "  <directory>     Directory containing .s files to scan"
    echo
    echo "Example:"
    echo "  $0 riscv-output"
    exit 1
}

# Show help if no argument or help flag is passed
if [[ -z "$1" || "$1" == "-h" || "$1" == "--help" ]]; then
    show_help
fi

dir="$1"

# Ensure the directory exists
if [[ ! -d "$dir" ]]; then
    echo "Error: Directory '$dir' not found"
    exit 1
fi

# Search for vector instructions in all .s files in the specified directory
results=$(grep -hoP '^\s*\Kv\w+' "$dir"/*.s 2>/dev/null | sort | uniq -c | sort -nr)

# Print the instruction counts if any
if [[ -n "$results" ]]; then
    echo "$results"
    echo
else
    echo "(no vector instructions found)"
    echo
fi

# Compute and print total
total=$(echo "$results" | awk '{s+=$1} END {print s+0}')
echo "Total = $total"
