import time
import numpy as np

from convolve import create_line_psf
from deblur import computeLocalPrior, updatePsi, computeL, updatef, save_mask_as_image
from helpers import open_image, write_image, kernel_from_image
MAX_ITER = 30
VARS = {
    'gamma': 2, # First iteration, then double
    'lambda1': 0.5, # [0.002, 0.5]
    'k1': 1.1, # [1.1, 1.4]
    'lambda2': 25, # [10, 25]
    'k2': 1.5,
}

# You can alter this parameters

# Open blurred picasso image
I = np.atleast_3d(open_image("examples/picassoBlurImage.png"))
# Create inicial kernel
f = create_line_psf(-np.pi/4, 1, (27, 27))
# Extracted rows, min(width, height)/2 - 12 for example
n_rows = 260

# Initialize Latent image with observed image I
L = I.copy() 
nL = I.copy() 

# Compute Omega region with t = 5
O_THRESHOLD = 5
s = time.time()
M = np.zeros_like(I)
for i in range(I.shape[2]):
    M[:, :, i] = computeLocalPrior(I[:, :, i], f.shape, O_THRESHOLD)
    save_mask_as_image(M[:, :, i], f"picasso_lp{i}.png")
print(f"computeLocalPrior took {time.time() - s}s")

# Calculate the observed image gradients for each channel
I_d = [np.gradient(I[:, :, i], axis=(1, 0)) for i in range(I.shape[2])]

# Initialize Psi
Psi = [[np.zeros(L.shape[:2]), np.zeros(L.shape[:2])] for _ in range(L.shape[2])]
nPsi = [[np.zeros(L.shape[:2]), np.zeros(L.shape[:2])] for _ in range(L.shape[2])]

tempao = 0
iterations = 0
while iterations < MAX_ITER:
    VARS['gamma'] = 2
    delta = 5000
    iters = 0
    while iters < 1:
        s = time.time()
        for i in range(L.shape[2]):
            L_d = np.gradient(L[:, :, i], axis=(1, 0))
            nPsi[i] = updatePsi(I_d[i], L_d, M[:, :, i], VARS['lambda1'], VARS['lambda2'], VARS['gamma'])
            nL[:, :, i] = computeL(L[:, :, i], I[:, :, i], f, nPsi[i], VARS['gamma'])
        deltaL = nL - L
        delta = np.linalg.norm(deltaL)
        print(delta)
        L = nL.copy()
        nPsi = Psi.copy()
        VARS['gamma'] *= 2
        iters += 1
    write_image(f'results/picasso{iterations}.png', L.copy())
    write_image(f'results/picasso_kernel{iterations}.png', f.copy()*(255/np.max(f)))
    f = updatef(L, I, f, n_rows=n_rows, k_cut_ratio=0)
    tempao_atual = time.time() - s
    tempao += tempao_atual
    print(f'{iterations}: {tempao}s')
    VARS['lambda1'] /= VARS['k1']
    VARS['lambda2'] /= VARS['k2']
    iterations += 1