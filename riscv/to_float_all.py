import struct
import numpy as np

def hex_to_float(hex_val):
    return struct.unpack('!f', bytes.fromhex(hex_val))[0]


def extract_vfredsum_values(filename):
    floats = []
    with open(filename, 'r') as file:
        for line in file:
            if "fsw" in line:
                parts = line.strip().split()
                hex_string = parts[6]
                if len(hex_string) >= 8:
                    last_8 = hex_string[-8:]
                    try:
                        floats.append(hex_to_float(last_8))
                    except Exception as e:
                        print(f"Skipping invalid hex '{last_8}': {e}")
    return floats

# Load and validate data
filename = "riscv/build/logs/dense.txt"
values = extract_vfredsum_values(filename)

# if len(values) != 24 * 24 * 8:
#     raise ValueError(f"Expected 4608 values, got {len(values)}")

# Convert to numpy array and reshape
# array = np.array(values, dtype=np.float32).reshape((24, 24, 8))
# array = np.array(values, dtype=np.float32).reshape((8, 24, 24))  # Shape: (channels, height, width)

for value in values:
    print(f"{value:.3f}", end=" ")

# for channel in range(8):
#     print(f"\nChannel {channel}:")
#     for i in range(24):
#         row = ""
#         for j in range(24):
#             row += f"{array[channel][i][j]:8.3f} "
#         print(row)