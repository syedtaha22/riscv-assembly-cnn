#!/bin/bash

# Help message function
show_help() {
    echo "Usage: $0 [-d <directory> | -f <file>] [--force-dis]"
    echo
    echo "Counts RISC-V vector instructions (e.g., vadd, vle, etc.) in .s or .dis files."
    echo
    echo "Options:"
    echo "  -d <directory>     Directory containing .s or .dis files to scan"
    echo "  -f <file>          Single .s or .dis file to scan"
    echo "  --force-dis        Treat all files as .dis (disassembly format)"
    echo "  -h, --help         Show this help message"
    echo
    echo "Examples:"
    echo "  $0 -d riscv-output"
    echo "  $0 -f riscv-output/main.s"
    echo "  $0 -d riscv-output --force-dis"
    exit 1
}

force_dis=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -d)
            target_type="dir"
            target_path="$2"
            shift 2
            ;;
        -f)
            target_type="file"
            target_path="$2"
            shift 2
            ;;
        --force-dis)
            force_dis=1
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# Validate input
if [[ -z "$target_type" || -z "$target_path" ]]; then
    show_help
fi

# Select files based on input type
if [[ "$target_type" == "dir" ]]; then
    if [[ ! -d "$target_path" ]]; then
        echo "Error: Directory '$target_path' not found"
        exit 1
    fi
    files=("$target_path"/*.{s,dis})
elif [[ "$target_type" == "file" ]]; then
    if [[ ! -f "$target_path" ]]; then
        echo "Error: File '$target_path' not found"
        exit 1
    fi
    files=("$target_path")
fi

# Set grep pattern depending on file type or forced dis mode
grep_results=""
for file in "${files[@]}"; do
    if [[ $force_dis -eq 1 || "$file" == *.dis ]]; then
        pattern='\s+\Kv(?:[a-z]+\.)?[a-z]+'
    else
        pattern='^\s*\Kv\w+'
    fi
    grep_results+=$(grep -hoP "$pattern" "$file" 2>/dev/null)
    grep_results+=$'\n'
done

# Process and output results
results=$(echo "$grep_results" | grep -v '^$' | sort | uniq -c | sort -nr)

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
