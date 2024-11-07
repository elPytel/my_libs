#By Pytel

import numpy as np

def shift(fft) -> np.ndarray:
    # np.fft.fftshift
    Y = fft.shape[0]
    X = fft.shape[1]
    ret = np.zeros([Y,X], dtype=complex)
    for y in range(Y):
        for x in range(X):
            ret[y][x] = fft[y][x]
    for y in range(Y):
        for x in range(X):
            ret[y][x] = fft[(y+Y//2)%Y][(x+X//2)%X]
    return ret

# amplitudové spektrum
def amp_spec(fft) -> np.ndarray:
    """ Calculate amplitude spectrum of image
    
    Args:
        fft : image to calculate amplitude spectrum
        
    Returns:
        amp_spec : amplitude spectrum of image
    """
    return np.log(np.abs(fft))