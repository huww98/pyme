import numpy as np

import pyme

def test_num_blocks():
    f = np.empty((128, 128), dtype=np.uint8)
    me = pyme.ESA(f, search_range=16)
    assert me.num_blocks(f) == [8, 8]

def test_set_blocking_offset():
    f = np.empty((128, 128), dtype=np.uint8)
    me = pyme.ESA(f, search_range=16)
    off = [8, 8]
    me.blocking_offset = off
    assert me.blocking_offset == off
    assert me.num_blocks(f) == [7, 7]
