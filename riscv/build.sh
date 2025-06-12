#!/bin/bash

# ================= Configuration =================
GCC_PREFIX="riscv32-unknown-elf"
ABI="-march=rv32gcv -mabi=ilp32d"
LINK="veer/link.ld"
WHISPER_CFG="veer/whisper.json"
BUILD_DIR="build"
OUT_DIRS=("exe" "hex" "dis" "logs")
# =================================================

show_help() {
    echo "Usage: $0 [options] <file.s> [<file.s> ...]"
    echo
    echo "Options:"
    echo "  -a         Compile and execute"
    echo "  -c         Clean generated files"
    echo "  -e         Execute the last compiled binary"
    echo "  -h         Show this help message"
    echo "  -l <file>  Link additional assembly files"
    echo
    echo "Example:"
    echo "  $0 -a main.s -l conv2d.s -l dense.s"
}

make_dirs() {
    for dir in "${OUT_DIRS[@]}"; do
        mkdir -p "${BUILD_DIR}/${dir}"
    done
}

get_basename() {
    filename="$1"
    echo "$(basename "$filename" .s)"
}

compile() {
    input_files=()
    # Extract files and add to the source files list
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -l)
                shift
                input_files+=("$1")  # Add the file after -l to the source files
                ;;
            *)
                input_files+=("$1")  # Add the file to the source files
                ;;
        esac
        shift
    done

    # Check if at least one file is provided
    if [[ ${#input_files[@]} -eq 0 ]]; then
        echo "Error: No files provided to compile."
        exit 1
    fi

    # Create file names based on input files
    base=$(get_basename "${input_files[0]}")
    exe="${BUILD_DIR}/exe/${base}.exe"
    hex="${BUILD_DIR}/hex/${base}.hex"
    dis="${BUILD_DIR}/dis/${base}.dis"
    data="${BUILD_DIR}/dis/${base}.data"

    echo "[*] Compiling files: ${input_files[*]} ..."

    # Compile and link all the files together
    $GCC_PREFIX-gcc $ABI -lgcc -T"$LINK" -o "$exe" "${input_files[@]}" -nostartfiles -lm

    # Generate output formats
    $GCC_PREFIX-objcopy -O verilog "$exe" "$hex"
    $GCC_PREFIX-objdump -S "$exe" > "$dis"
    $GCC_PREFIX-objdump -s -j .data "$exe" > "$data"

    echo "[+] Output: $exe, $hex, $dis, $data"
}


execute() {
    input_file="$1"
    if [[ ! -f "$input_file" ]]; then
        echo "Error: $input_file not found."
        exit 1
    fi

    base=$(get_basename "$input_file")
    hex_file="${BUILD_DIR}/hex/${base}.hex"
    log_file="${BUILD_DIR}/logs/${base}.txt"

    if [[ ! -f "$hex_file" ]]; then
        echo "Error: $hex_file not found. Compile first."
        exit 1
    fi

    echo "[*] Executing with whisper..."
    whisper -x "$hex_file" -s 0x80000000 --tohost 0xd0580000 -f "$log_file" --configfile "$WHISPER_CFG"
    echo "[+] Execution log saved to $log_file"
}

clean() {
    echo "[*] Cleaning generated files..."
    rm -rf "$BUILD_DIR"
    echo "[+] Clean complete."
}

# ===================== Main ======================
if [[ $# -eq 0 ]]; then
    show_help
    exit 1
fi

ACTION=""

while getopts "aceh" opt; do
    case $opt in
        a) ACTION="all" ;;
        c) ACTION="clean" ;;
        e) ACTION="exec" ;;
        h) show_help; exit 0 ;;
        *) show_help; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

make_dirs

case "$ACTION" in
    all)
        if [[ $# -lt 1 ]]; then
            echo "Error: Please provide at least one .s file."
            show_help
            exit 1
        fi
        compile "$@"
        execute "$1"
        ;;
    clean)
        clean
        ;;
    exec)
        if [[ $# -lt 1 ]]; then
            echo "Error: Please provide a .s file for execution."
            show_help
            exit 1
        fi
        execute "$1"
        ;;
    *)
        show_help
        ;;
esac
