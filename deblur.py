""" Implementation of 'High-Quality Motion Deblurring from a Single Image (SIGGRAPH 2008)'
    Eduardo Stuani
    Gustavo Prolla Lacroix

    NOTE:
    * use 64bit double precision for I, L and f
    * see approach for processing boundaries of Liu and Jia [2008]
"""
import time
import cv2
import numpy as np
from scipy.fft import fft2, ifft2
from scipy.ndimage import gaussian_filter

### LOAD IMAGE ###
image_path = 'examples/picassoBlurImage.png'
image_cv = cv2.imread(image_path)
# image_cv_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB) # Might not be needed, must check
I = np.array(image_cv, np.float64)

L = I.copy() # Latent image

# Initialize f as a 2D Gaussian kernel using scipy's gaussian_filter
k_size = 27
sigma = 5
f = np.zeros((k_size, k_size))
f[k_size // 2, k_size // 2] = 1  # Set the center pixel to 1
f = gaussian_filter(f, sigma=sigma)
# Normalize the kernel to ensure that the sum of all elements is 1
f /= np.sum(f)


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
    # Initialize M to be the same size as I
    M = np.zeros_like(I, dtype=np.uint8)
    std_dev = np.zeros(3)

    # Iterate through pixels
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            # Extract local window centered at (i, j) with size defined by f
            top = max(0, i - f[0] // 2)
            bottom = min(I.shape[0], i + f[0] // 2 + 1)
            left = max(0, j - f[1] // 2)
            right = min(I.shape[1], j + f[1] // 2 + 1)

            window = I[top:bottom, left:right]

            # Calculate STD of pixel colors in each window
            cv2.meanStdDev(window, None, std_dev)

            # Check if std < t, then set M[i, j] to 1, else set it to 0
            M[i, j] = np.all(std_dev < t)
    return M

def save_mask_as_image(mask, output_path):
    mask_image = (mask * 255).astype(np.uint8)
    cv2.imwrite(output_path, mask_image)

stime = time.time()
M = computeLocalPrior(I, f.shape, O_THRESHOLD)
etime = time.time()
runtime = etime - stime
print(f"computeLocalPrior took {runtime}s")
save_mask_as_image(M, "picassoBlurMask.png")
exit(0)

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
