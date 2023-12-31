import cv2
import numpy as np
from scipy.fft import fft2, ifft2, ifftshift, fftshift
from scipy.signal import convolve2d


def create_line_psf(theta, scale, sz):
    """
    Create a Point Spread Function (PSF) in the form of a straight line.

    Parameters:
    - theta: angle of the line in radians
    - scale: float, in the interval [0, 1]
    - sz: tuple, the desired size of the PSF

    Returns:
    - psf: 2D array, the PSF

    TODO:
    - remove division by zero
    """
    psf = np.zeros(sz)
    X = sz[1] // 2
    Y = sz[0] // 2
    theta = theta % np.pi

    # Calcular os pontos de intersecao
    if theta <= np.pi/2:
        p1 = (min(Y/np.tan(theta), X), min(X*np.tan(theta), Y))
    else:
        p1 = (max(Y/np.tan(theta), -X), min(-X*np.tan(theta), Y))

    # Calcular os pontos escalados
    p1 = (int(p1[0]*scale), int(p1[1]*scale))
    p2 = (-p1[0], -p1[1])

    # Mudanca de coordenada
    p1 = (p1[0]+X, -p1[1]+Y)
    p2 = (p2[0]+X, -p2[1]+Y)

    thickness = int(np.mean(sz) * 0.005)
    thickness = 1 if thickness < 1 else thickness
    psf = cv2.line(psf, p2, p1, color=255, thickness=thickness, lineType=cv2.LINE_AA)
    return psf


def psf2otf(psf, sz):
    """
    Compute the FFT of the Point Spread Function (PSF) and pad/crop it so
    it has the shape specified by sz.

    Parameters:
    - psf: 2D array, the Point Spread Function.
    - sz: tuple, the desired size of the output OTF.

    Returns:
    - otf: 2D array, the Optical Transfer Function.

    TODO:
    Add case when psf is bigger than the image
    """
    psf = np.atleast_2d(psf)
    psf_sz = psf.shape

    if psf_sz != sz:
        # Pad PSF with zeros to match specified dimensions (sz)
        diffx = sz[0] - psf_sz[0]
        diffy = sz[1] - psf_sz[1]
        diffx2 = diffx // 2
        diffy2 = diffy // 2
        restx = diffx % 2
        resty = diffy % 2
        psf = np.pad(psf, [(diffx2, diffx2+restx), (diffy2, diffy2+resty)], mode='constant')    

    otf = fft2(fftshift(psf))
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
