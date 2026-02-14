from unittest.mock import MagicMock

import numpy as np

from smrf.data import Topo

SKY_VIEW_FACTOR_MOCK = np.array([[1.0, 1.0]])
TOPO_MOCK = MagicMock(
    spec=Topo,
    sky_view_factor=SKY_VIEW_FACTOR_MOCK,
    veg_height=np.array([[5.0, 10.0]]),
    veg_k=np.array([[0.8, 0.1]]),
    veg_tau=np.array([[0.6, 0.7]]),
    instance=True,
)
