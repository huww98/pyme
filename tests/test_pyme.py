import pytest
import numpy as np

import pyme

def test_block_size():
    assert pyme.ESA.block_size == 16

def test_num_blocks():
    f = np.empty((128, 128), dtype=np.uint8)
    me = pyme.ESA(f, search_range=16)
    assert me.num_blocks(f) == [8, 8]

def test_zero_mv():
    f = np.zeros((128, 128), dtype=np.uint8)
    me = pyme.ESA(f, search_range=16)
    mv = me.estimate(f)
    zero_mv = np.mgrid[:128:me.block_size, :128:me.block_size].transpose(1, 2, 0)
    assert (mv == zero_mv).all()

def test_small_ref_img():
    ref = np.zeros((8, 8), dtype=np.uint8)
    me = pyme.ESA(ref, search_range=16)
    cur = np.zeros((128, 128), dtype=np.uint8)
    mv = me.estimate(cur)
    assert (mv == -1).all()

def test_small_cur_img():
    ref = np.zeros((128, 128), dtype=np.uint8)
    me = pyme.ESA(ref, search_range=16)
    cur = np.zeros((8, 8), dtype=np.uint8)
    mv = me.estimate(cur)
    assert mv.shape == (0,0,2)

@pytest.mark.parametrize(['offset', 'expected_mv'] ,[
    (16, 0),
    (17, -1),
])
@pytest.mark.parametrize('dir', [1, -1])
@pytest.mark.parametrize('axis', ['x', 'y'])
def test_large_offset(offset, expected_mv, dir, axis):
    offset *= dir
    offset = (offset, 0) if axis == 'x' else (0, offset)
    ref = np.zeros((16, 16), dtype=np.uint8)
    me = pyme.ESA(ref, search_range=16, ref_offset=offset)
    cur = np.zeros((16, 16), dtype=np.uint8)
    mv = me.estimate(cur)
    assert mv.shape == (1,1,2)
    assert (mv == expected_mv).all()
