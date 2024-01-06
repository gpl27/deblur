"""
Implementation of 'High-Quality Motion Deblurring from a Single Image (SIGGRAPH 2008)'

    Eduardo Stuani
    Gustavo Prolla Lacroix

    NOTE:
    * Works with numba version 0.58.1
    * Each channel of the image is processed independently, with the exception
      of the local prior mask M, which is calculated using the standard
      deviation of all channels.
"""
import cv2
import warnings
import numpy as np
from scipy.fft import fft2, ifft2
from numba import njit
from numba.core.errors import NumbaPendingDeprecationWarning
from convolve import psf2otf 


# Filter Numba deprecation warnings
warnings.filterwarnings("ignore", category=NumbaPendingDeprecationWarning)

def get_derivatives(matrix):
    """
    Calculates all derivatives of matrix from the set theta 
    of derivative operators.

    Parameters:
    - matrix: 2D array

    Returns
    - derivatives: dictionary containg the derivatives
    """
    derivatives = {
        'd0': matrix.copy(),
        'dx': np.gradient(matrix, axis=1),
        'dy': np.gradient(matrix, axis=0),
        'dxx': np.gradient(np.gradient(matrix, axis=1), axis=1),
        'dyy': np.gradient(np.gradient(matrix, axis=0), axis=0),
        'dxy': np.gradient(np.gradient(matrix, axis=1), axis=0),
        'dyx': np.gradient(np.gradient(matrix, axis=0), axis=1),
    }
    return derivatives

def computeLocalPrior(I, f, t):
    """
    Compute the local prior M for each pixel in I. The standard deviation
    is calculated for each channel. All channels must be below the threshold
    t for the pixel to be considered in the local prior.

    Parameters:
    - I: 3D array, the input image (observed image) with shape (height, width, channels)
    - f: 2D array, the PSF or filter kernel
    - t: float, threshold for the standard deviation

    Returns:
    - M: 2D array, the local prior mask with shape (height, width)
    """
    I = np.atleast_3d(I)
    M = np.zeros(I.shape[:2], dtype=np.uint8)
    std_dev = np.zeros(I.shape[2])

    # Iterate through pixels
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            # Extract local window centered at (i, j) with size defined by f
            top = max(0, i - f[0] // 2)
            bottom = min(I.shape[0], i + f[0] // 2 + 1)
            left = max(0, j - f[1] // 2)
            right = min(I.shape[1], j + f[1] // 2 + 1)
            window = I[top:bottom, left:right]

            cv2.meanStdDev(window, None, std_dev)
            M[i, j] = np.all(std_dev < t)
    return M

def save_mask_as_image(mask, output_path):
    """
    Saves a mask as an image.

    Parameters:
    - mask: 2D array, the mask to be saved
    - output_path: string, the path to save the image
    """
    mask_image = (mask * 255).astype(np.uint8)
    cv2.imwrite(output_path, mask_image)

@njit
def updatePsi(I_d, L_d, M, lambda1, lambda2, gamma):
    """
    Updates the Psi values for a single channel. By the paper definition,
    Psi ~ L_d. Therefore, gamma is used to weight the latent image gradient
    and is increased at each iteration.

    Parameters:
    - I_d: 3D array, the observed image gradient with shape (direction, height, width)
    - L_d: 3D array, the latent image gradient with shape (direction, height, width)
    - M: 2D array, the local prior mask with shape (height, width)
    - lambda1: float, the weight for the global prior
    - lambda2: float, the weight for the local prior
    - gamma: float, the weight for the latent image gradient

    Returns:
    - nPsi: 3D array, the updated Psi with shape (direction, height, width)
    """
    k = 2.7
    a = 6.1e-4
    b = 5.0
    lt = 1.852
    x = np.zeros(3)
    nPsi = [np.zeros_like(M), np.zeros_like(M)]
    for v in range(2):
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                x1 = (lambda2*M[i, j]*I_d[v][i, j] + gamma*L_d[v][i, j])/(-a*lambda1 + lambda2*M[i, j] + gamma)
                x[0] = x1 if x1 > lt or x1 < -lt else np.NAN
                x2 = ((k/2)*lambda1 + lambda2*M[i, j]*I_d[v][i, j] + gamma*L_d[v][i, j])/(lambda2*M[i, j] + gamma)
                x[1] = x2 if x2 >= -lt and x2 < 0 else np.NAN
                x3 = ((-k/2)*lambda1 + lambda2*M[i, j]*I_d[v][i, j] + gamma*L_d[v][i, j])/(lambda2*M[i, j] + gamma)
                x[2] = x3 if x3 >= 0 and x3 <= lt else np.NAN
                result = np.nanmin(x)
                nPsi[v][i, j] = result if not np.isnan(result) else L_d[v][i, j]
    return nPsi

def computeL(L, I, f, Psi, gamma):
    """
    Compute the latent image L for a single channel.

    Parameters:
    - L: 2D array, the latent image with shape (height, width)
    - I: 2D array, the observed image with shape (height, width)
    - f: 2D array, the PSF or filter kernel
    - Psi: 3D array, the latent image gradient with shape (direction, height, width)
    - gamma: float, the weight for the latent image gradient

    Returns:
    - L_star: 2D array, the updated latent image with shape (height, width)
    """
    # Derivatives and derivative weights for the Delta calculation
    d = get_derivatives(L)
    d_w = {
        'd0': 0,
        'dx': 1,
        'dy': 1,
        'dxx': 2,
        'dyy': 2,
        'dxy': 2,
        'dyx': 2
    }

    # Calculate Delta
    delta = np.zeros(I.shape, np.complex128)
    for key in d:
        dv = fft2(d[key])
        delta += np.complex128(50/(2**d_w[key]))*np.conjugate(dv)*dv

    # Calculate L*
    F_f = psf2otf(f, I.shape)
    CF_f = np.conjugate(F_f)
    F_dx = fft2(d['dx'])
    CF_dx = np.conjugate(F_dx)
    F_dy = fft2(d['dy'])
    CF_dy = np.conjugate(F_dy)
    L_nominator = CF_f*fft2(I)*delta + gamma*(CF_dx*fft2(Psi[0])) + gamma*(CF_dy*fft2(Psi[1]))
    L_denominator = CF_f*F_f*delta + gamma*(CF_dx*F_dx) + gamma*(CF_dy*F_dy)
    L_star = L_nominator / L_denominator
    L_star = ifft2(L_star).real.astype(np.float64)
    return L_star

def updatef():
    """
    TODO
    """
    pass
