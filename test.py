import time
import numpy as np

from deblur import computeLocalPrior, updatePsi, computeL, updatef, save_mask_as_image
from helpers import open_image, write_image, kernel_from_image


def test_with_picasso():
    VARS = {
        'gamma': 2, # First iteration, then double
        'lambda1': 0.002, # [0.002, 0.5]
        'k1': 1.1, # [1.1, 1.4]
        'lambda2': 10, # [10, 25]
        'k2': 1.5,
    }

    # Open blurred picasso image
    I = open_image("examples/picassoBlurImage.png")

    # Initialize Latent image with observed image I
    L = I.copy() 

    # Open the kernel que criei no pixelart xDDDD
    f = kernel_from_image('examples/kerneeeeeeeeeeel.png')

    # Compute Omega region with t = 5
    O_THRESHOLD = 5
    s= time.time()
    M = computeLocalPrior(I, f.shape, O_THRESHOLD)
    print(f"computeLocalPrior took {time.time() - s}s")
    save_mask_as_image(M, "picasso_lp.png")

    # Calculate the observed image gradients for each channel
    I_d = [np.gradient(I[:, :, i], axis=(1, 0)) for i in range(I.shape[2])]
    
    # Initialize Psi
    Psi = [[np.zeros(L.shape[:2]), np.zeros(L.shape[:2])] for _ in range(L.shape[2])]

    # Alternate between updating Psi and L for each channel
    iterations = 0
    MAX_ITERATIONS = 15
    VARS['gamma'] = 2
    while iterations < MAX_ITERATIONS:
        s = time.time()
        for i in range(L.shape[2]):
            L_d = np.gradient(L[:, :, i], axis=(1, 0))
            Psi[i] = updatePsi(I_d[i], L_d, M, VARS['lambda1'], VARS['lambda2'], VARS['gamma'])
            L[:, :, i] = computeL(L[:, :, i], I[:, :, i], f, Psi[i], VARS['gamma'])
        f = updatef(L, I, f)
        VARS['gamma'] *= 2
        write_image(f'picasso{iterations}.png', L)
        print(f'{iterations}: {time.time() - s}s')
        iterations += 1


test_with_picasso()
