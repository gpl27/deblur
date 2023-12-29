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
from scipy.optimize import minimize


# Partial derivative operators
# Use np.diff(matrix, axis=1) : this calculates dx
# Use np.diff(matrix, axis=0) : this calculates dy
# Use np.diff(np.diff(matrix, axis=1), axis=1) : this calculates dxx
# Use np.diff(np.diff(matrix, axis=0), axis=0) : this calculates dyy
# Use np.diff(np.diff(matrix, axis=1), axis=0) : this calculates dxy

VARS = {
    'gamma': 2, # First iteration, then double
    'lambda1': 0.002, # [0.002, 0.5]
    'k1': 1.1, # [1.1, 1.4]
    'lambda2': 10, # [10, 25]
    'k2': 1.5,
}

# Function Phi is an approximation of the logarithmic gradient distribution
def Phi(x, lt):
    k = 2.7
    a = 6.1e-4
    b = 5.0
    return np.piecewise(x, [x <= lt, x > lt], [lambda x: -k * np.abs(x), lambda x: -(a * x**2 + b)])

# Function E'psi_i,v is used to minimize each element of Psi independently to form a global minimum
def Epsi(psi, Phi, lambda1, lambda2, Mi, gamma, dIi, dLi):
    return lambda1*np.abs(Phi(psi)) + lambda2*Mi*((psi - dIi)**2) + gamma*((psi - dLi)**2)


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

def updatePsi(Psi, I, L, M, VARS):
    I_v = (np.diff(I, axis=1), np.diff(I, axis=0))
    L_v = (np.diff(L, axis=1), np.diff(L, axis=0))
    for v, Psi_v in enumerate(Psi):
        for i in range(Psi.size[1]):
            for j in range(Psi.size[1]):
                initial_psi_i_v = Psi_v[i][j]
                result = minimize(Epsi, initial_psi_i_v, args=(Phi, VARS['lambda1'], VARS['lambda2'], M[i][j], VARS['gamma'], I_v[v][i][j], L_v[v][i][j]))
                Psi_v[i][j] = result.x

def computeL(I, L, f, VARS):
    # Calculate Delta

    # Calculate L*

    pass

def updatef():
    pass

### ALGORITHM ###

image_path = 'examples/picassoBlurImage.png'
image_cv = cv2.imread(image_path)

# image_cv_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB) # Might not be needed, must check
I = np.array(image_cv, np.float64)

# Initialize L with observed image I
L = I.copy() # Latent image

# Initialize f as a 2D Gaussian kernel using scipy's gaussian_filter
k_size = 27
sigma = 5
f = np.zeros((k_size, k_size))
f[k_size // 2, k_size // 2] = 1  # Set the center pixel to 1
f = gaussian_filter(f, sigma=sigma)
# Normalize the kernel to ensure that the sum of all elements is 1
f /= np.sum(f)

# Compute Omega region with t = 5
O_THRESHOLD = 5 # Omega Region Threshold
M = computeLocalPrior(I, f, O_THRESHOLD) # Binary map

iterations = 0
MAX_ITERATIONS = 20
ERROR = 10**-5

stime = time.time()
M = computeLocalPrior(I, f.shape, O_THRESHOLD)
etime = time.time()
runtime = etime - stime
print(f"computeLocalPrior took {runtime}s")
save_mask_as_image(M, "picassoBlurMask.png")

# Psi[0] = Psi_x ; Psi[1] = Psi_y
Psi = [np.diff(I, axis=1), np.diff(I, axis=0)]

while 1 > ERROR and iterations < MAX_ITERATIONS: # Optimizing L and f
    # Optimize L
    VARS['gamma'] = 2
    while 1 > ERROR or 1 > ERROR:
        updatePsi(Psi, I, L, M, VARS)
        computeL()
        VARS['gamma'] *= 2
    # Optimife f
    updatef()

# Save L and f
