from convolve import *
from helpers import *
import numpy as np

image = open_image("examples/CM.png")
psf = create_line_psf(np.deg2rad(45), 0.05, image.shape[:2])
motion_blur = convolve_in_frequency_domain(image, psf)
write_image("CMmotion.png", motion_blur)
