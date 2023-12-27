""" Implementation of 'High-Quality Motion Deblurring from a Single Image (SIGGRAPH 2008)'
    Eduardo Stuani
    Gustavo Prolla Lacroix

    NOTE:
    * use 64bit double precision for I, L and f
    * see approach for processing boundaries of Liu and Jia [2008]
"""
import cv2
import numpy as np
from scipy.fft import fft2, ifft2

I = np.mat() # Blurred image
L = np.mat() # Latent image
f = np.mat() # Blur kernel or PSF

O_THRESHOLD = 5 # Omega Region Threshold

THETA = set() # Partial derivative operators
def pdx():
    pass

VARS = {
    'gama': 2, # First iteration, then double
    'lambda1': 0.002, # [0.002, 0.5]
    'k1': 1.1, # [1.1, 1.4]
    'lambda2': 10, # [10, 25]
    'k2': 1.5,
}

### METHODS ###
def computeLocalPrior(I, f, t):
    # f is just to define the size of the local window
    # Initialize M to be the same size as I

    # Iterate through pixels
        # Calculate STD of pixel colors in each window
        # if std < t, then M[i] = 1 else 0
    return M

def updatePsi():
    pass

def computeL():
    pass

def updatef():
    pass

### ALGORITHM ###
# Compute Omega region with t = 5
M = computeLocalPrior(I, f, O_THRESHOLD) # Binary map

# Initialize L with observed image I

iterations = 0
MAX_ITERATIONS = 20
ERROR = 10**-5
while 1 > ERROR and iterations < MAX_ITERATIONS: # Optimizing L and f
    # Optimize L
    while 1 > ERROR or 1 > ERROR:
        updatePsi()
        computeL()
    # Optimife f
    updatef()

# Save L and f
