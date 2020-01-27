import numpy as np
import xxhash


class ArrayWrapper:

    def __init__(self, array):
        self.array = array
        self.hash_val = hash_output(self.array)

    def __hash__(self):
        # print(self.hash_val)
        return self.hash_val

    def __eq__(self, other):
        return self.__class__ == other.__class__ and np.array_equal(self.array, other.array)

    def __setattr__(self, attr, value):
        if hasattr(self, attr):
            raise Exception("Attempting to alter read-only value")

        self.__dict__[attr] = value


def hash_output(dat_array):
    x = xxhash.xxh64_intdigest(dat_array.data.tobytes(), seed=0)
    # print(x)
    # hashOutput = hash(dat_array.data.tobytes())
    return x
    # return hashOutput