import json
import numpy as np
import torch


def get_hparams_from_file(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    return hparams

class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


def string_to_bits(string, pad_len=8):
    # Convert each character to its ASCII value
    ascii_values = [ord(char) for char in string]
    
    # Convert ASCII values to binary representation
    binary_values = [bin(value)[2:].zfill(8) for value in ascii_values]
    
    # Convert binary strings to integer arrays
    bit_arrays = [[int(bit) for bit in binary] for binary in binary_values]
    
    # Convert list of arrays to NumPy array
    numpy_array = np.array(bit_arrays)
    numpy_array_full = np.zeros((pad_len, 8), dtype=numpy_array.dtype)
    numpy_array_full[:, 2] = 1
    max_len = min(pad_len, len(numpy_array))
    numpy_array_full[:max_len] = numpy_array[:max_len]
    return numpy_array_full


def bits_to_string(bits_array):
    # Convert each row of the array to a binary string
    binary_values = [''.join(str(bit) for bit in row) for row in bits_array]
    
    # Convert binary strings to ASCII values
    ascii_values = [int(binary, 2) for binary in binary_values]
    
    # Convert ASCII values to characters
    output_string = ''.join(chr(value) for value in ascii_values)
    
    return output_string