import struct
import argparse

FEATURE_SET_HASH = 0x7f234cb8
VERSION = 0x7AF32F20
DEFAULT_DESCRIPTION = "Network trained with the https://github.com/official-stockfish/nnue-pytorch trainer."

def encode_leb_128_array(arr):
    res = []
    for v in arr:
        while True:
            byte = v & 0x7f
            v = v >> 7
            if (v == 0 and byte & 0x40 == 0) or (v == -1 and byte & 0x40 != 0):
                res.append(byte)
                break
            res.append(byte | 0x80)
    return res

# def fc_hash(L1, L2, L3, num_ls_buckets=8):
#     # InputSlice hash
#     prev_hash = 0xEC42E90D
#     prev_hash ^= (L1 * 2)

#     # Fully connected layers
#     layers_info = [
#         (L2 + 1, True), 
#         (L3, True),      
#         (1, False)      
#     ]
    
#     for out_features, has_relu in layers_info:
#         layer_hash = 0xCC03DAE4
#         layer_hash += out_features // num_ls_buckets
#         layer_hash ^= prev_hash >> 1
#         layer_hash ^= (prev_hash << 31) & 0xFFFFFFFF
#         if out_features // num_ls_buckets != 1 and has_relu:
#             # Clipped ReLU hash
#             layer_hash = (layer_hash + 0x538D24C7) & 0xFFFFFFFF
#         prev_hash = layer_hash
    
#     return prev_hash

def write_header(outfile, L1, fc_hash_val, description=DEFAULT_DESCRIPTION):
    # Write version
    outfile.write(struct.pack('<I', VERSION))
    
    # Write network hash
    # halfkp_hash = fc_hash_val ^ FEATURE_SET_HASH ^ (L1 * 2)
    halfkp_hash = 0x1C1020F2
    outfile.write(struct.pack('<I', halfkp_hash))
    
    # Write description
    encoded_description = description.encode('utf-8')
    outfile.write(struct.pack('<I', len(encoded_description)))
    outfile.write(encoded_description)

def write_leb_128_array(outfile, arr):
    buf = encode_leb_128_array(arr)
    outfile.write(struct.pack('<I', len(buf)))
    outfile.write(bytes(buf))

def read_binary_file(filename, L1, L2, L3, inputs, num_buckets=8):
    with open(filename, 'rb') as f:
        data = {}
        
        # Read l0b (16-bit) - 3072 values
        data['l0b'] = list(struct.unpack('<' + 'h' * 3072, f.read(3072 * 2)))

        # Read l0w (16-bit) - 69206016 values (inputs * L1)
        data['l0w'] = list(struct.unpack('<' + 'h' * (inputs * L1), f.read(inputs * L1 * 2)))

        # Read pst (32-bit) - 720896 values (inputs * 8)
        data['pst'] = list(struct.unpack('<' + 'i' * (inputs * num_buckets), f.read(inputs * num_buckets * 4)))

        # Read l1b (32-bit) - 128 values ((L2 + 1) * 8)
        data['l1b'] = list(struct.unpack('<' + 'i' * ((L2 + 1) * num_buckets), f.read((L2 + 1) * num_buckets * 4)))
        
        # Read l1w (8-bit) - 393216 values (L1 * (L2 + 1) * 8)
        data['l1w'] = list(struct.unpack('<' + 'b' * (L1 * (L2 + 1) * num_buckets), f.read(L1 * (L2 + 1) * num_buckets)))
        
        # Read l2b (32-bit) - 256 values (L3 * 8)
        data['l2b'] = list(struct.unpack('<' + 'i' * (L3 * num_buckets), f.read(L3 * num_buckets * 4)))
        
        # Read l2w (8-bit) - 7680 values ((L2 + 1) * 2 * L3 * 8)
        data['l2w'] = list(struct.unpack('<' + 'b' * ((L2 + 1) * 2 * L3 * num_buckets), f.read((L2+1) * 2 * L3 * num_buckets)))
        
        # Read l3b (32-bit) - 8 values (8)
        data['l3b'] = list(struct.unpack('<' + 'i' * num_buckets, f.read(num_buckets * 4)))
        
        # Read l3w (8-bit) - 256 values (L3 * 8)
        data['l3w'] = list(struct.unpack('<' + 'b' * (L3 * num_buckets), f.read(L3 * num_buckets)))
        
        return data

def organize_into_buckets(data, L1, L2, L3, num_buckets=8):
    bucketed_data = {
        'l0b': data['l0b'],  # No bucketing for l0b
        'l0w': data['l0w'],  # No bucketing for l0w
        'pst': data['pst'],  # Already bucketed as inputs * 8
        'l1': [],
        'l2': [],
        'l3': []
    }

    # Organize l1 layer (bias and weights) into buckets
    for bucket in range(num_buckets):
        bias_start = bucket * (L2 + 1)
        bias_end = bias_start + (L2 + 1)
        weights_start = bucket * L1 * (L2 + 1)
        weights_end = weights_start + L1 * (L2 + 1)
        
        bucketed_data['l1'].append({
            'bias': data['l1b'][bias_start:bias_end],
            'weights': data['l1w'][weights_start:weights_end]
        })
    
    # Organize l2 layer (bias and weights) into buckets
    for bucket in range(num_buckets):
        bias_start = bucket * L3
        bias_end = bias_start + L3
        weights_start = bucket * (L2 + 1) * 2 * L3
        weights_end = weights_start + (L2 + 1) * 2 * L3

        weights = data['l2w'][weights_start:weights_end]
        
        bucketed_data['l2'].append({
            'bias': data['l2b'][bias_start:bias_end],
            'weights': weights
        })

    # Organize l3 layer (bias and weights) into buckets
    for bucket in range(num_buckets):
        bias_start = bucket
        bias_end = bias_start + 1
        weights_start = bucket * L3
        weights_end = weights_start + L3
        
        bucketed_data['l3'].append({
            'bias': data['l3b'][bias_start:bias_end],
            'weights': data['l3w'][weights_start:weights_end]
        })
    
    return bucketed_data

def convert_binary_format(input_file, output_file, L1=3072, L2=15, L3=32, inputs=22528, num_buckets=8):
    # Read the input binary file
    data = read_binary_file(input_file, L1, L2, L3, inputs, num_buckets)

    print(f"Read {input_file} successfully.")
    
    # Organize data into buckets
    bucketed_data = organize_into_buckets(data, L1, L2, L3, num_buckets)

    print(f"Organized data into {num_buckets} buckets.")
    
    # Calculate hashes
    # fc_hash_val = fc_hash(L1, L2, L3, num_buckets)

    print(f"Writing to {output_file}...")
    
    with open(output_file, 'wb') as outfile:
        # Write header
        write_header(outfile, L1, "unused")
        
        # Write feature transformer hash
        feature_transformer_hash = FEATURE_SET_HASH ^ (L1 * 2)
        outfile.write(struct.pack('<I', feature_transformer_hash))
        
        # Write l0 bias, weights and psqt in LEB128 format
        outfile.write('COMPRESSED_LEB128'.encode('utf-8'))
        write_leb_128_array(outfile, bucketed_data['l0b'])
        
        outfile.write('COMPRESSED_LEB128'.encode('utf-8'))
        write_leb_128_array(outfile, bucketed_data['l0w'])
        
        outfile.write('COMPRESSED_LEB128'.encode('utf-8'))
        write_leb_128_array(outfile, bucketed_data['pst'])
        
        # Write each bucket of l1, l2 and l3 layers
        fc_hash_val = 0x6333744A;
        for bucket in range(num_buckets):
            outfile.write(struct.pack('<I', fc_hash_val))
            
            assert len(bucketed_data['l1'][bucket]['bias']) == (L2 + 1)
            assert len(bucketed_data['l1'][bucket]['weights']) == L1 * (L2 + 1)

            outfile.write(struct.pack('<' + 'i' * len(bucketed_data['l1'][bucket]['bias']), 
                                     *bucketed_data['l1'][bucket]['bias']))
            outfile.write(struct.pack('<' + 'b' * len(bucketed_data['l1'][bucket]['weights']), 
                                     *bucketed_data['l1'][bucket]['weights']))
            
            # start_position = outfile.tell()
            
            assert len(bucketed_data['l2'][bucket]['bias']) == L3
            assert len(bucketed_data['l2'][bucket]['weights']) == (L2 + 1) * 2 * L3 
            
            outfile.write(struct.pack('<' + 'i' * len(bucketed_data['l2'][bucket]['bias']), 
                                     *bucketed_data['l2'][bucket]['bias']))
            outfile.write(struct.pack('<' + 'b' * len(bucketed_data['l2'][bucket]['weights']), 
                                     *bucketed_data['l2'][bucket]['weights']))
            
            # end_position = outfile.tell()
            # bucket_size = end_position - start_position
            # print(f"Ending position for bucket {bucket}: {end_position}")
            # print(f"Bucket {bucket} size: {bucket_size} bytes")

            assert len(bucketed_data['l3'][bucket]['bias']) == 1
            assert len(bucketed_data['l3'][bucket]['weights']) == L3

            outfile.write(struct.pack('<' + 'i' * len(bucketed_data['l3'][bucket]['bias']), 
                                     *bucketed_data['l3'][bucket]['bias']))
            outfile.write(struct.pack('<' + 'b' * len(bucketed_data['l3'][bucket]['weights']), 
                                     *bucketed_data['l3'][bucket]['weights']))



def main():
    parser = argparse.ArgumentParser(description='Convert binary neural network format')
    parser.add_argument('input_file', type=str, help='Input binary file path')
    parser.add_argument('output_file', type=str, help='Output binary file path')
    parser.add_argument('--L1', type=int, default=3072, help='L1 parameter (default: 3072)')
    parser.add_argument('--L2', type=int, default=15, help='L2 parameter (default: 15)')
    parser.add_argument('--L3', type=int, default=32, help='L3 parameter (default: 32)')
    parser.add_argument('--inputs', type=int, default=22528, help='inputs parameter (default: 22528)')
    parser.add_argument('--buckets', type=int, default=8, help='Number of buckets (default: 8)')
    
    args = parser.parse_args()
    
    convert_binary_format(args.input_file, args.output_file, args.L1, args.L2, args.L3, args.inputs, args.buckets)
    test_foo(args.input_file, args.output_file, args.L1, args.L2, args.L3, args.inputs, args.buckets)
    print(f"Conversion complete: {args.input_file} -> {args.output_file}")

def test_foo(input_file, output_file, L1=3072, L2=15, L3=32, inputs=22528, num_buckets=8):
     with open(output_file, 'rb') as f:
        # data = f.read()

        f.seek(69439819)

        four_bytes = f.read(4)

        integer_value = struct.unpack('<I', four_bytes)[0]
        print(f"Integer value at position 69389475: {integer_value}")


if __name__ == "__main__":
    main()