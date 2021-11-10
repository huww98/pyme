import numpy as np

from . import _C

class ESA(_C.ESA):
    def estimate(self, cur_frame, mv=None):
        if mv is None:
            mv = np.empty(self.num_blocks(cur_frame), dtype=np.int32)
        return super().estimate(cur_frame, mv)
