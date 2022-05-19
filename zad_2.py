import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import string

from skimage.feature import peak_local_max

sensitivity = 0.03
letter_size = 0.62

space_len = 50
line_height = 32

filename = "./test.PNG"
font_path = "./CourierPrime"

text = np.array(Image.open(filename).convert('L'))
text = 255 - text

k = 300
#U, E, VT = np.linalg.svd(text)
#text = np.zeros((len(U), len(VT)), dtype=float)
#for i in range(k):
#    text += E[i] * np.outer(U.T[i], VT[i])

#plt.imshow(text, cmap='Greys')
#plt.show()

letters = list(string.ascii_lowercase)
letter_coords = []

for l in letters:
    f = font_path + "/" + l + ".png"
    letter = Image.open(f).convert('L')
    letter = letter.resize((int(letter.width*letter_size), int(letter.height*letter_size)))
    letter = np.array(letter)
    letter = 255 - letter

    #plt.imshow(letter)
    #plt.show()

    C = np.real(np.fft.ifft2(np.fft.fft2(text) * np.fft.fft2(np.rot90(letter, 2), text.shape)))

    coordinates = peak_local_max(C, min_distance=10, threshold_rel=1-sensitivity)

    for i in range(len(coordinates)):
        letter_coords.append((coordinates[i][0], coordinates[i][1], l))

    #plt.imshow(text, cmap='Greys')
    #plt.plot(coordinates[:, 1], coordinates[:, 0], 'r.')
    #plt.show()


for i in range(line_height, text.shape[0], line_height):
    for j in range(len(letter_coords)):
        if i > letter_coords[j][0] >= i-line_height:
            letter_coords[j] = (i-line_height, letter_coords[j][1], letter_coords[j][2])

letter_coords.sort()
output = open("output.txt",'w')
prev_y = letter_coords[0][0]
prev_x = letter_coords[0][1]
for y, x, letter in letter_coords:
    if y-prev_y >= line_height:
        output.write("\n")
    elif x-prev_x >= space_len:
        output.write(" ")
    output.write(letter)
    prev_x = x
    prev_y = y

output.close()