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
    echo "Usage: $0 [options] <file.s>"
    echo
    echo "Options:"
    echo "  -a         Compile and execute"
    echo "  -c         Clean generated files"
    echo "  -e         Execute the last compiled binary"
    echo "  -h         Show this help message"
    echo
    echo "Example:"
    echo "  $0 -a vvaddint32.s"
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
    input_file="$1"
    if [[ ! -f "$input_file" ]]; then
        echo "Error: $input_file not found."
        exit 1
    fi

    base=$(get_basename "$input_file")
    exe="${BUILD_DIR}/exe/${base}.exe"
    hex="${BUILD_DIR}/hex/${base}.hex"
    dis="${BUILD_DIR}/dis/${base}.dis"

    echo "[*] Compiling $input_file ..."
    $GCC_PREFIX-gcc $ABI -lgcc -T"$LINK" -o "$exe" "$input_file" -nostartfiles -lm
    $GCC_PREFIX-objcopy -O verilog "$exe" "$hex"
    $GCC_PREFIX-objdump -S "$exe" > "$dis"
    echo "[+] Output: $exe, $hex, $dis"
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

while getopts "aceh" opt; do
    case $opt in
        a) ACTION="all" ;;
        c) ACTION="clean" ;;
        e) ACTION="exec" ;;
        h) show_help; exit 0 ;;
        *) show_help; exit 1 ;;
    esac
done
shift $((OPTIND -1))

make_dirs

case "$ACTION" in
    all)
        if [[ $# -ne 1 ]]; then
            echo "Error: Please provide a .s file."
            show_help
            exit 1
        fi
        compile "$1"
        execute "$1"
        ;;
    clean)
        clean
        ;;
    exec)
        if [[ $# -ne 1 ]]; then
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
