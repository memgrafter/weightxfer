"""weightxfer: robust skin weight transfer (Abdrashitov et al., SIGGRAPH Asia 2023) on numpy, scipy and
robust-laplacian. MIT licence; the method's reference code is MIT, Copyright (c) 2024 Rinat Abdrashitov (NOTICE)."""
from .closest import closest_points
from .geometry import adjacency, vertex_normals
from .inpaint import inpaint, laplacian, smooth
from .limit import chain_bones, limit_influences, normalize, restrict_bones
from .transfer import match, transfer

__version__ = "0.1.0"
__all__ = ["transfer", "match", "inpaint", "laplacian", "smooth", "limit_influences", "normalize", "chain_bones",
           "restrict_bones", "closest_points", "vertex_normals", "adjacency"]
