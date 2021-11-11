import pytest
import numpy as np

import pyme

def test_block_size():
    assert pyme.ESA.block_size == 16

def test_ref_offset_prop():
    f = np.empty((128, 128), dtype=np.uint8)
    me = pyme.ESA(f, search_range=16, ref_offset=(8,8))
    assert me.ref_offset == [8,8]

def test_num_blocks():
    f = np.empty((128, 128), dtype=np.uint8)
    me = pyme.ESA(f, search_range=16)
    assert me.num_blocks(f) == [8, 8]

def test_zero_mv():
    f = np.zeros((128, 128), dtype=np.uint8)
    me = pyme.ESA(f, search_range=16)
    mv, cost = me.estimate(f)
    zero_mv = np.mgrid[:128:me.block_size, :128:me.block_size].transpose(1, 2, 0)
    assert (mv == zero_mv).all()

def test_search_to_zero_mv():
    ref = np.zeros((32, 32), dtype=np.uint8)
    ref[1, 3] = 222
    me = pyme.ESA(ref, search_range=16)

    cur = np.zeros((16, 16), dtype=np.uint8)
    cur[1, 3] = 199
    mv, cost = me.estimate(cur)
    assert (mv == 0).all()

def test_me():
    ref = np.zeros((32, 32), dtype=np.uint8)
    ref[6,7] = 164
    me = pyme.ESA(ref, search_range=16)

    cur = np.zeros((32, 32), dtype=np.uint8)
    cur[1,1] = 88
    cur[19,21] = 203
    mv, cost = me.estimate(cur)

    expected_mv = np.array([
        [[5,  6], [0, 16]],
        [[16, 0], [3,  2]],
    ])
    expected_cost = np.array([
        [164-88, 0],
        [0, 203-164],
    ])

    assert (mv == expected_mv).all()
    assert (cost == expected_cost).all()

def test_small_ref_img():
    ref = np.zeros((8, 8), dtype=np.uint8)
    me = pyme.ESA(ref, search_range=16)
    cur = np.zeros((128, 128), dtype=np.uint8)
    mv, cost = me.estimate(cur)
    assert (mv == -1).all()
    assert (cost == np.iinfo(cost.dtype).max).all()

def test_small_cur_img():
    ref = np.zeros((128, 128), dtype=np.uint8)
    me = pyme.ESA(ref, search_range=16)
    cur = np.zeros((8, 8), dtype=np.uint8)
    mv, cost = me.estimate(cur)
    assert mv.shape == (0,0,2)
    assert cost.shape == (0,0)

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
    mv, cost = me.estimate(cur)
    assert mv.shape == (1,1,2)
    assert (mv == expected_mv).all()
