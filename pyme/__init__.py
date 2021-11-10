import numpy as np

from . import _C

class ESA(_C.ESA):
    def estimate(self, cur_frame, mv=None):
        if mv is None:
            mv = np.empty(self.num_blocks(cur_frame) + [2], dtype=np.int32)
        super().estimate(cur_frame, mv)
        return mv
