import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import scipy.ndimage as ndimage

sensitivity = 5

galia = np.array(Image.open("Lab9_galia.png"))
galia = 255-galia
imp = np.array(Image.open("Lab9_galia_e.png"))
imp = 255-imp

C = np.real(np.fft.ifft2(np.fft.fft2(rgb2gray(galia)) * np.fft.fft2(np.rot90(rgb2gray(imp), 2), (463, 953))))

#https://stackoverflow.com/questions/9111711/get-coordinates-of-local-maxima-in-2d-array-above-certain-value

indices = np.nonzero(C > np.amax(C)-sensitivity)
x, y = [], []
for i in range(len(indices[0])):
    y.append(indices[0][i])
    x.append(indices[1][i])


plt.imshow(galia)
plt.plot(x,y, 'ro')

plt.show()
