import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
sensitivity = 0.05

galia = np.array(Image.open("Lab9_galia.png").convert('L'))
galia = 255-galia
imp = np.array(Image.open("Lab9_galia_e.png").convert('L'))
imp = 255-imp

plt.imshow(galia, cmap='Greys')
plt.show()
plt.imshow(imp, cmap='Greys')
plt.show()

C = np.real(np.fft.ifft2(np.fft.fft2(galia) * np.fft.fft2(np.rot90(imp, 2), (463, 953))))

indices = np.nonzero(C > np.amax(C)*(1-sensitivity))
x, y = [], []
for i in range(len(indices[0])):
    y.append(indices[0][i])
    x.append(indices[1][i])


plt.imshow(galia, cmap='Greys')
plt.plot(x,y, 'ro')

plt.show()
