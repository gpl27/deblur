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

from convolve import psf2otf, create_line_psf, convolve_in_frequency_domain, convolve_in_spatial_domain, add_gaussian_noise
from helpers import kernel_from_image, write_image, open_image




# Function Phi is an approximation of the logarithmic gradient distribution
def Phi(x):
    k = 2.7
    a = 6.1e-4
    b = 5.0
    lt = 1.852
    return np.piecewise(x, [x <= lt, x > lt], [lambda x: -k * np.abs(x), lambda x: -(a * x**2 + b)])

# Function E'psi_i,v is used to minimize each element of Psi independently to form a global minimum
def Epsi(psi, lambda1, lambda2, Mi, gamma, dIi, dLi):
    return lambda1*Phi(np.abs(psi)) + lambda2*Mi*((psi - dIi)**2) + gamma*((psi - dLi)**2)


### METHODS ###
def get_derivatives(matrix):
    """
    Calculates all the derivates from the set theta of the
    provided matrix.

    Parameters:
    - matrix: 2D array

    Returns
    - derivatives: dictionary containg the derivatives

    NOTE:
    Partial derivative operators:
    np.diff(matrix, axis=1) : this calculates dx
    np.diff(matrix, axis=0) : this calculates dy
    np.diff(np.diff(matrix, axis=1), axis=1) : this calculates dxx
    np.diff(np.diff(matrix, axis=0), axis=0) : this calculates dyy
    np.diff(np.diff(matrix, axis=1), axis=0) : this calculates dxy

    """
    derivatives = {
        'dx': np.gradient(matrix, axis=1),
        'dy': np.gradient(matrix, axis=0),
        'dxx': np.gradient(np.gradient(matrix, axis=1), axis=1),
        'dyy': np.gradient(np.gradient(matrix, axis=0), axis=0),
        'dxy': np.gradient(np.gradient(matrix, axis=1), axis=0),
    }
    return derivatives

def delta_2D_M(A, B):
    """
    Calculates the the 2-norm of A-B
    """
    delta = np.abs(A - B)
    return np.linalg.norm(delta, ord=2)

def computeLocalPrior(I, f, t):
    # Initialize M to be the same size as I
    M = np.zeros_like(I, dtype=np.uint8)
    std_dev = np.zeros(1)

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
    I_d = (np.gradient(I, axis=1), np.gradient(I, axis=0))
    L_d = (np.gradient(L, axis=1), np.gradient(L, axis=0))
    for v, Psi_v in enumerate(Psi):
        for i in range(Psi_v.shape[0]):
            for j in range(Psi_v.shape[1]):
                initial_psi_i_v = Psi_v[i, j]
                result = minimize(Epsi, initial_psi_i_v, args=(VARS['lambda1'], VARS['lambda2'], M[i, j], VARS['gamma'], I_d[v][i, j], L_d[v][i, j]), method='L-BFGS-B')
                Psi_v[i, j] = result.x[0]


def W(d):
    d_w = {
        'dx': 1,
        'dy': 1,
        'dxx': 2,
        'dyy': 2,
        'dxy': 2
    }
    return np.complex128(50/(2**d_w[d]))


def computeL(L, I, f, Psi, VARS):
    d = get_derivatives(L)
    # Calculate Delta
    delta = np.zeros(I.shape, np.complex128)
    for key in d:
        dv = fft2(d[key])
        delta += W(key)*np.conjugate(dv)*dv
    # Calculate L*
    F_f = psf2otf(f, I.shape)
    CF_f = np.conjugate(F_f)
    F_dx = fft2(d['dx'])
    CF_dx = np.conjugate(F_dx)
    F_dy = fft2(d['dy'])
    CF_dy = np.conjugate(F_dy)
    L_nominator = CF_f*fft2(I)*delta + VARS['gamma']*CF_dx*fft2(Psi[0]) + VARS['gamma']*CF_dy*fft2(Psi[1])
    L_denominator = CF_f*F_f*delta + VARS['gamma']*CF_dx*F_dx + VARS['gamma']*CF_dy*F_dy
    L_star = L_nominator / L_denominator
    L_star = ifft2(L_star).astype(np.float64)
    return L_star


def updatef():
    pass

### ALGORITHM ###

def test_with_CM():
    VARS = {
        'gamma': 2, # First iteration, then double
        'lambda1': 0.002, # [0.002, 0.5]
        'k1': 1.1, # [1.1, 1.4]
        'lambda2': 10, # [10, 25]
        'k2': 1.5,
    }

    VARS['lambda1'] /= VARS['k1']**10
    VARS['lambda2'] /= VARS['k2']**10

    I = cv2.imread('examples/CM.png', flags=cv2.IMREAD_GRAYSCALE)
    I = np.array(I, np.float64)
    psf = create_line_psf(np.deg2rad(45), 0.5, (27, 27))
    I = np.squeeze(add_gaussian_noise(convolve_in_spatial_domain(I, psf)).astype(np.float64), axis=-1)
    write_image("CMmotion.png", I)
    f = psf.copy()

    # Compute Omega region with t = 5
    O_THRESHOLD = 5 # Omega Region Threshold
    stime = time.time()
    M = computeLocalPrior(I, f.shape, O_THRESHOLD)
    etime = time.time()
    runtime = etime - stime
    print(f"computeLocalPrior took {runtime}s")
    save_mask_as_image(M, "CM_lp.png")

    # Initialize L with observed image I
    I = I / np.max(I)
    L = I.copy() # Latent image

    iterations = 0
    MAX_ITERATIONS = 15
    VARS['gamma'] = 2
    Psi = [np.zeros_like(L), np.zeros_like(L)]
    # For the time being I am ignoring the deltas
    while iterations < MAX_ITERATIONS:
        s = time.time()
        updatePsi(Psi, I, L, M, VARS)
        L = computeL(L, I, f, Psi, VARS)
        VARS['gamma'] *= 2
        factor = 255/np.max(L)
        write_image(f'CM{iterations}.png', (L*factor).astype(np.uint8))
        print(f'{iterations}: {time.time() - s}s')
        iterations += 1


def test_with_picasso():
    VARS = {
        'gamma': 2, # First iteration, then double
        'lambda1': 0.002, # [0.002, 0.5]
        'k1': 1.1, # [1.1, 1.4]
        'lambda2': 10, # [10, 25]
        'k2': 1.5,
    }

    VARS['lambda1'] /= (VARS['k1']**10)
    VARS['lambda2'] /= (VARS['k2']**10)

    I = cv2.imread('examples/picassoBlurImage.png', flags=cv2.IMREAD_GRAYSCALE)
    I = np.array(I, np.float64)
    psf = kernel_from_image('examples/picassoBlurImage_kernel.png')

    # Initialize L with observed image I
    L = I.copy() # Latent image
    f = psf.copy()

    # Compute Omega region with t = 5
    O_THRESHOLD = 5 # Omega Region Threshold
    stime = time.time()
    M = computeLocalPrior(I, f.shape, O_THRESHOLD)
    etime = time.time()
    runtime = etime - stime
    print(f"computeLocalPrior took {runtime}s")
    save_mask_as_image(M, "Picasso_lp.png")

    iterations = 0
    MAX_ITERATIONS = 15

    VARS['gamma'] = 2
    Psi = [np.gradient(L, axis=1), np.gradient(L, axis=0)]
    # For the time being I am ignoring the deltas
    while iterations < MAX_ITERATIONS:
        s = time.time()
        # updatePsi(Psi, I, L, M, VARS)
        L = computeL(L, I, f, Psi, VARS)
        VARS['gamma'] *= 2
        factor = 255/np.max(L)
        write_image(f'picasso{iterations}.png', (L*factor).astype(np.uint8))
        print(f'{iterations}: {time.time() - s}s')
        iterations += 1

test_with_picasso()
