"""SFCNet-CPC model package.

Convenience re-exports so callers can do `from models import NetMultiClass`.
"""
from .NetMultiClass import NetMultiClass       # SFCNet-CPC  (main model)
from .NetFS_CPC import NetFS_CPC               # FS-CPC variant
from .NetCC_CPC import NetCC_CPC               # CA-CPC variant
from .Net import Net                           # SFCNet binary backbone (for reference)
from .smt import smt_t                         # SMT-Tiny encoder

__all__ = ["NetMultiClass", "NetFS_CPC", "NetCC_CPC", "Net", "smt_t"]
