import numpy as np

from . import _C

class ESA(_C.ESA):
    def estimate(self, cur_frame, mv=None, cost=None):
        if mv is None or cost is None:
            num_blocks = self.num_blocks(cur_frame)
            if mv is None:
                mv = np.empty(num_blocks + [2], dtype=np.int32)
            if cost is None:
                cost = np.empty(num_blocks, dtype='=Q')
        super().estimate(cur_frame, mv, cost)
        return mv, cost
