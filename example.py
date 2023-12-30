from convolve import *
from helpers import *

path_unblurred = "examples/picassoOut.png"
path_blurred = 'convolvefftpadded.png'
kernel_path = "examples/picassoBlurImage_kernel.png"

unblurred = open_image(path_unblurred)
kernel = kernel_from_image(kernel_path)

blurred_s = convolve_in_spatial_domain(unblurred, kernel)
write_image("blurred_s.png", blurred_s)
blurred_f = convolve_in_frequency_domain(unblurred, kernel)
write_image("blurred_f.png", blurred_f)

deconvolved_s = deconvolve(blurred_s, kernel)
write_image("deconv_s.png", deconvolved_s)
deconvolved_f = deconvolve(blurred_f, kernel)
write_image("deconv_f.png", deconvolved_f)
