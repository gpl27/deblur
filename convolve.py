import numpy as np
from scipy.fft import fft2, ifft2, ifftshift
from scipy.signal import convolve2d

def psf2otf(psf, sz):
    """
    Compute the FFT of the Point Spread Function (PSF) and pad/crop it so
    it has the shape specified by sz.

    Parameters:
    - psf: 2D array, the Point Spread Function.
    - sz: tuple, the desired size of the output OTF.

    Returns:
    - otf: 2D array, the Optical Transfer Function.
    """
    psf = np.atleast_2d(psf)
    psf_sz = psf.shape

    # Pad PSF with zeros to match specified dimensions (sz)
    psf_padded = np.pad(psf, [(0, sz[0] - psf_sz[0]), (0, sz[1] - psf_sz[1])], mode='constant')

    # Circularly shift PSF so that the central pixel is at (1, 1) position
    shift_amnt = ((sz[0] - psf_sz[0])//2, (sz[1] - psf_sz[1])//2)
    psf_shifted = np.roll(psf_padded, shift=shift_amnt, axis=(0, 1))

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

    NOTE:
    Will add option for padding in the future. The problem with padding is that
    you can't simply deconvolve the image with the same kernel and restore it
    """
    psf = np.atleast_2d(psf)
    num_channels = image.shape[2]
    convolved_image = np.zeros_like(image, dtype=np.complex128)
    otf_shape = (image.shape[0], image.shape[1])
    otf = psf2otf(psf, otf_shape)

    # Perform convolution in the frequency domain for each channel
    for channel in range(num_channels):
        image_fft = fft2(image[:, :, channel])
        result_fft = image_fft * otf
        convolved_image[:, :, channel] = np.real(ifft2(result_fft))
    
    return convolved_image

def convolve_in_spatial_domain(image, kernel):
    """
    Convolve a 3-channel image with a kernel in the spatial domain.

    Parameters:
    - image: 3D array, the input image with shape (height, width, channels)
    - kernel: 2D array, the kernel or PSF

    Returns:
    - convolved_image: 3D array, the convolved image with same orginal shape
    """
    convolved_image = np.zeros_like(image)
    for v in range(image.shape[2]):
        convolved_image[:, :, v] = convolve2d(image[:, :, v], kernel, mode='same', boundary='symm')
    return convolved_image

def deconvolve(image, psf):
    """
    Deconvolve a 3-channel image with a point spread function (PSF)

    Parameters:
    - image: 3D array, the input image with shape (height, width, channels).
    - psf: 2D array, the point spread function.

    Returns:
    - deconvolved_image: 3D array, the deconvolved image with shape (height, width, channels).

    NOTE:
    See `convolve_in_frequency_domain` for notes about padding
    """
    num_channels = image.shape[2]
    deconvolved_image = np.zeros_like(image, dtype=np.complex128)
    otf_shape = (image.shape[0], image.shape[1])
    otf = psf2otf(psf, otf_shape)

    # Perform deconvolution in the frequency domain for each channel
    for channel in range(num_channels):
        image_fft = fft2(image[:, :, channel])
        result_fft = image_fft / otf
        deconvolved_image[:, :, channel] = np.real(ifft2(result_fft))
    
    return deconvolved_image
