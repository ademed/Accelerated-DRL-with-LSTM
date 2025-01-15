import numpy as np

def spherecov(h, L):
    r = h/L
    if r < 1:
        s = (1 - (3/2)*r + (1/2)*np.power(r, 3))
    else:
        s = 0

    return s