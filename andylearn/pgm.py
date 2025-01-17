import numpy as np
import struct

def loadPGM(name):
    original_data_file = open(name,"rb")
    data_buff = original_data_file.read()
    data_index = 0
    s = struct.unpack_from('>14c',data_buff, data_index)
    data_index += struct.calcsize('>14c')
    m = struct.unpack_from('>10304B',data_buff, data_index)
    m = np.mat(m)
    return m

