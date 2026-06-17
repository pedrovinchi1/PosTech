import os
import platform
import random

# macOS: torch e xgboost trazem runtimes OpenMP (libomp) próprios; quando ambos
# são carregados no mesmo processo o programa pode sofrer segmentation fault.
# Limitar o OpenMP a 1 thread no macOS evita o conflito (sem efeito em Win/Linux).
# Precisa ser definido antes de importar torch/numpy.
if platform.system() == "Darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
