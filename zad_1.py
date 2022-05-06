import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv

galia = np.array(Image.open("Lab9_galia.png"))
galia = 255-galia
imp = np.array(Image.open("Lab9_galia_e.png"))
imp = 255-imp

C = np.real(np.fft.ifft2(np.fft.fft2(rgb2gray(galia)) * np.fft.fft2(np.rot90(rgb2gray(imp),2), (463, 953))))
plt.plot(C)
plt.show()
