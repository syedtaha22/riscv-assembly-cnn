import struct

# Input hex string
hex_string = "36c100ca32392c883f7ffe84b11263833120e69eb10e63862f4bfdd0378d6034"

# Ensure hex string has a length that's a multiple of 8 characters (4 bytes)
if len(hex_string) % 8 != 0:
    hex_string = hex_string.rstrip()

# Split the hex string into chunks of 8 characters (4 bytes each)
chunks = [hex_string[i:i+8] for i in range(0, len(hex_string), 8)]

# Function to convert hex to float
def hex_to_float(hex_value):
    # Convert hex to bytes
    bytes_value = bytes.fromhex(hex_value)
    # Convert bytes to float (assuming IEEE 754 standard)
    return struct.unpack('!f', bytes_value)[0]

# Convert each chunk to its corresponding float value
float_values = [hex_to_float(chunk) for chunk in chunks]

# three decimal places
float_values = [round(value, 3) for value in float_values]

# reverse the list to get the original order
float_values.reverse()

# Print the resulting float values
print(float_values)
