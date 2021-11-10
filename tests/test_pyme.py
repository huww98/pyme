import numpy as np

import pyme

def test_block_size():
    assert pyme.ESA.block_size == 16

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

def test_zero_mv():
    f = np.zeros((128, 128), dtype=np.uint8)
    me = pyme.ESA(f, search_range=16)
    mv = me.estimate(f)
    zero_mv = np.mgrid[:128:me.block_size, :128:me.block_size].transpose(1, 2, 0)
    assert (mv == zero_mv).all()
