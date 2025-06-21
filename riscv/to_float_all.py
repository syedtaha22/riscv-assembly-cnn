import struct
import numpy as np

def hex_to_float(hex_val, use_float16=False):
    # Ensure even-length hex string
    if len(hex_val) % 2 != 0:
        hex_val = "0" + hex_val  # pad left if needed

    byte_data = bytes.fromhex(hex_val)
    
    if use_float16:
        if len(byte_data) != 2:
            raise ValueError(f"Invalid half-precision hex: {hex_val}")
        # Swap for little-endian if needed
        byte_data = byte_data[::-1]
        return struct.unpack('<e', byte_data)[0]
    else:
        if len(byte_data) != 4:
            raise ValueError(f"Invalid single-precision hex: {hex_val}")
        byte_data = byte_data[::-1]
        return struct.unpack('<f', byte_data)[0]

def extract_floats_from_log(filename, instruction="vlse16", first_only=False, use_float16=False):
    floats = []
    chunk_size = 4 if use_float16 else 8  # hex digits: 4 for f16, 8 for f32

    with open(filename, 'r') as file:
        for line in file:
            if instruction in line:
                parts = line.strip().split()
                if len(parts) < 7:
                    continue
                hex_string = parts[6]

                # Handle scalar register (e.g., f3 = 8 hex digits)
                if len(hex_string) == 8:
                    try:
                        if use_float16:
                            # Use only least significant 16 bits (last 4 hex digits)
                            floats.append(hex_to_float(hex_string[4:], use_float16=True))
                        else:
                            floats.append(hex_to_float(hex_string, use_float16=False))
                    except Exception as e:
                        print(f"Skipping invalid scalar hex: {e}")
                    continue

                # Otherwise treat as packed vector register
                chunks = [hex_string[i:i + chunk_size] for i in range(0, len(hex_string), chunk_size)]
                chunks.reverse()  # memory order correction

                try:
                    if first_only:
                        floats.append(hex_to_float(chunks[0], use_float16))
                    else:
                        for chunk in chunks:
                            floats.append(hex_to_float(chunk, use_float16))
                except Exception as e:
                    print(f"Skipping invalid hex chunk: {e}")
    return floats

def display_floats(values, shape=None, channel_view=False):
    if shape is not None:
        try:
            array = np.array(values, dtype=np.float32).reshape(shape)
        except ValueError as e:
            raise ValueError(f"Could not reshape: {e}")

        if channel_view and len(shape) == 3:
            channels, height, width = shape
            for c in range(channels):
                print(f"\nChannel {c}:")
                for i in range(height):
                    row = " ".join(f"{array[c][i][j]:8.3f}" for j in range(width))
                    print(row)
        else:
            print(array)
    else:
        print("Flat output:")
        for value in values:
            print(f"{value:.3f}", end=" ")
        print()

if __name__ == "__main__":
    filename = "build/logs/main.txt"
    instruction = "vle16.v v10, (a0)"

    values = extract_floats_from_log(
        filename,
        instruction=instruction,
        first_only=False,
        use_float16=True  # Change this to True if you're parsing float16
    )

    # values = values[1::2]  # Skip every second value
    values = values[:10]  

    display_floats(
        values,
        # shape=None,  # Set to None for flat output
        shape=None,
        channel_view=True
    )

    # Print model prediction
    print("\nModel prediction:", np.argmax(values))
