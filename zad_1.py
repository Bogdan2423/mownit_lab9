import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

sensitivity = 0.05

galia = np.array(Image.open("Lab9_galia.png").convert('L'))
galia = 255-galia
imp = np.array(Image.open("Lab9_galia_e.png").convert('L'))
imp = 255-imp

C = np.real(np.fft.ifft2(np.fft.fft2(galia) * np.fft.fft2(np.rot90(imp, 2), galia.shape)))

indices = np.nonzero(C > np.amax(C)*(1-sensitivity))
x, y = [], []
for i in range(len(indices[0])):
    y.append(indices[0][i])
    x.append(indices[1][i])

letter_num = len(x)

plt.imshow(galia, cmap='Greys')
plt.plot(x,y, 'ro')

plt.show()


sensitivity = 0.20

school_img = Image.open("Lab9_school.jpg").convert('L')
school = np.array(school_img)
fish_img = school_img.crop((227, 296, 250, 320))
fish = np.array(fish_img)

plt.imshow(school, cmap='Greys')
plt.show()
plt.imshow(fish, cmap='Greys')
plt.show()

C = np.real(np.fft.ifft2(np.fft.fft2(school) * np.fft.fft2(np.rot90(fish, 2), school.shape)))

indices = np.nonzero(C > np.amax(C)*(1-sensitivity))
x, y = [], []
for i in range(len(indices[0])):
    y.append(indices[0][i])
    x.append(indices[1][i])

fish_num = len(x)

plt.imshow(school, cmap='Greys')
plt.plot(x,y, 'ro')

plt.show()