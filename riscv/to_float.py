import struct

# Input hex string
hex_string = "0000000000000000000000004800ec00a800340030008600b200b20090000400"

# Toggle one of these to True
use_fp16 = True
use_fp32 = False
use_int16 = False  # New flag for 16-bit signed integers

# Determine chunk size
chunk_size = 4 if use_fp32 else 2
hex_chunk_len = chunk_size * 2

# Truncate hex string to a multiple of chunk size
if len(hex_string) % hex_chunk_len != 0:
    hex_string = hex_string[:len(hex_string) - (len(hex_string) % hex_chunk_len)]

# Split hex string into chunks
chunks = [hex_string[i:i+hex_chunk_len] for i in range(0, len(hex_string), hex_chunk_len)]

# Conversion functions
def hex_to_float16(hex_value):
    bytes_value = bytes.fromhex(hex_value)
    half = int.from_bytes(bytes_value, byteorder='big')
    sign = ((half >> 15) & 0x00000001)
    exponent = ((half >> 10) & 0x0000001f)
    fraction = (half & 0x03ff)

    if exponent == 0:
        if fraction == 0:
            return float((-1)**sign * 0.0)
        else:
            return (-1)**sign * 2**(-14) * (fraction / 1024)
    elif exponent == 0x1F:
        if fraction == 0:
            return float('inf') if sign == 0 else float('-inf')
        else:
            return float('nan')
    else:
        return (-1)**sign * 2**(exponent - 15) * (1 + fraction / 1024)

def hex_to_float32(hex_value):
    bytes_value = bytes.fromhex(hex_value)
    return struct.unpack('!f', bytes_value)[0]

def hex_to_int16(hex_value):
    bytes_value = bytes.fromhex(hex_value)
    return struct.unpack('!h', bytes_value)[0]

# Convert
if use_fp32:
    float_values = [round(hex_to_float32(chunk), 10) for chunk in chunks]
elif use_fp16:
    float_values = [round(hex_to_float16(chunk), 10) for chunk in chunks]
elif use_int16:
    float_values = [hex_to_int16(chunk) for chunk in chunks]
else:
    raise ValueError("Enable one of use_fp16, use_fp32, or use_int16")

# Reverse for original order
float_values.reverse()

# Print result
print(float_values)
