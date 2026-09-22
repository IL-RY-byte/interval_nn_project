"""Interval-corridor neural networks (NumPy only).

Every weight W_ij is allowed to vary inside a relative corridor
    W_ij in [C_ij - s_l*|C_ij|, C_ij + s_l*|C_ij|],   s_l = rho * gamma_l
where gamma_l is the per-layer diversity coefficient. Training minimises the
worst-case loss over the whole corridor (and, optionally, an input box).
"""
