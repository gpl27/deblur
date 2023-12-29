import time
import cv2
import numpy as np
from scipy.fft import fft2, ifft2, ifftshift
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from scipy.signal import convolve2d

def open_image(image_path: str) -> np.ndarray:
    """Returns image as NumPy Array 64-bit float with 3 channels (RGB)"""
    image_cv = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)
    return np.array(image_cv, np.float64)

def psf2otf(psf, sz):
    """
    Compute the FFT of the Point Spread Function (PSF) and circularly shift
    it to ensure the central pixel is at (1, 1) position.

    Parameters:
    - psf: 2D array, the Point Spread Function.
    - sz: tuple, the desired size of the output OTF.

    Returns:
    - otf: 2D array, the Optical Transfer Function.
    """
    # Ensure psf is 2D array
    psf = np.atleast_2d(psf)

    # Get the size of the PSF
    psf_sz = psf.shape

    # Pad PSF with zeros to match specified dimensions (sz)
    psf_padded = np.pad(psf, [(0, sz[0] - psf_sz[0]), (0, sz[1] - psf_sz[1])], mode='constant')

    # Circularly shift PSF so that the central pixel is at (1, 1) position
    shift_amnt = ((sz[0] - psf_sz[0])//2, (sz[1] - psf_sz[1])//2)
    psf_shifted = np.roll(psf_padded, shift=shift_amnt, axis=(0, 1))

    # Compute the FFT of the PSF
    otf = fft2(ifftshift(psf_shifted))

    return otf


def convolve_in_frequency_domain(image, psf):
    """
    Convolve a 3-channel image with a point spread function (PSF) in the frequency domain.

    Parameters:
    - image: 3D array, the input image with shape (height, width, channels).
    - psf: 2D array, the point spread function.

    Returns:
    - convolved_image: 3D array, the convolved image with shape (height, width, channels).
    """
    # Ensure psf is 2D array
    psf = np.atleast_2d(psf)

    # Get the number of channels
    num_channels = image.shape[2]

    # Initialize the convolved image
    convolved_image = np.zeros_like(image, dtype=np.complex128)

    # TODO: make it so padding is not an arbritrary value
    pad = 10
    otf_shape = (image.shape[0]+2*pad, image.shape[1]+2*pad)
    otf = psf2otf(psf, otf_shape)

    padded_image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), 'symmetric')

    # Perform convolution in the frequency domain for each channel
    for channel in range(num_channels):
        # Apply Fourier Transform to the image and PSF
        image_fft = fft2(padded_image[:, :, channel])

        # Convolve in the frequency domain
        result_fft = image_fft * otf

        # Apply Inverse Fourier Transform and shift back
        convolved_image[:, :, channel] = np.real(ifft2(result_fft))[pad:-pad, pad:-pad]
    
    return convolved_image

def convolve_in_spatial_domain(image, kernel):
    result = np.zeros_like(image)
    for v in range(image.shape[2]):
        result[:, :, v] = convolve2d(image[:, :, v], kernel, mode='same', boundary='symm')
    return result

def kernel_from_image(image_path: str, shape) -> np.ndarray:
    image_cv = cv2.imread(image_path, flags=cv2.IMREAD_GRAYSCALE)
    kernel = np.array(image_cv)
    kernel = kernel / np.sum(kernel)
    return kernel


image_path = "examples/picassoOut.png"
kernel_path = "examples/picassoBlurImage_kernel.png"

image = open_image(image_path)
kernel = kernel_from_image(kernel_path, image.shape[:2])

c_image = convolve_in_frequency_domain(image, kernel)
cv2.imwrite("convolvefftpadded.png", c_image.astype(np.uint8))