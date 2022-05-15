import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.feature import peak_local_max


galia = np.array(Image.open("Lab9_galia.png").convert('L'))
galia = 255-galia
imp = np.array(Image.open("Lab9_galia_e.png").convert('L'))
imp = 255-imp

C = np.real(np.fft.ifft2(np.fft.fft2(galia) * np.fft.fft2(np.rot90(imp, 2), galia.shape)))

coordinates = peak_local_max(C, min_distance=20, threshold_rel=0.96)

plt.imshow(galia, cmap='Greys')
plt.plot(coordinates[:, 1], coordinates[:, 0], 'r.')
plt.show()


letter_num = len(coordinates)


school_img = Image.open("Lab9_school.jpg").convert('L')
school = np.array(school_img)
fish_img = school_img.crop((227, 296, 250, 320))
fish = np.array(fish_img)

C = np.real(np.fft.ifft2(np.fft.fft2(school) * np.fft.fft2(np.rot90(fish, 2), school.shape)))

coordinates = peak_local_max(C, min_distance=10, threshold_rel=0.70)

plt.imshow(school, cmap='Greys')
plt.plot(coordinates[:, 1], coordinates[:, 0], 'r.')
plt.show()

fish_num = len(coordinates)

print("Number of letters \'e\':",letter_num)
print("Number of fish:",fish_num)