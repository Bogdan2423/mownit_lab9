import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import string

sensitivity = 0.01
letter_size = 0.50

space_len = 40
line_height = 30

filename = "./test.PNG"
font_path = "./font"

text = np.array(Image.open(filename).convert('L'))
text = 255 - text

k = 300
U, E, VT = np.linalg.svd(text)
text = np.zeros((len(U), len(VT)), dtype=float)
for i in range(k):
    text += E[i] * np.outer(U.T[i], VT[i])

plt.imshow(text, cmap='Greys')
plt.show()

letters = list(string.ascii_lowercase)
letter_coords = []

for l in letters:
    f = font_path + "/" + l + ".png"
    letter = Image.open(f).convert('L')
    letter = letter.resize((int(letter.width*letter_size), int(letter.height*letter_size)))
    letter = np.array(letter)
    letter = 255 - letter

    plt.imshow(letter)
    plt.show()

    C = np.real(np.fft.ifft2(np.fft.fft2(text) * np.fft.fft2(np.rot90(letter, 2), text.shape)))


    indices = np.nonzero(C >= (1-sensitivity)*np.amax(C))

    for i in range(len(indices[0])):
        letter_coords.append((indices[0][i], indices[1][i], l))

    x, y = [], []
    for i in range(len(indices[0])):
        y.append(indices[0][i])
        x.append(indices[1][i])


    plt.imshow(text, cmap='Greys')
    plt.plot(x,y, 'ro')
    plt.show()


letter_coords.sort()

output = open("output.txt",'w')
prev_x = letter_coords[0][1]
prev_y = letter_coords[0][0]
for y, x, letter in letter_coords:
    if y-prev_y >= line_height:
        output.write("\n")
    elif x-prev_x >= space_len:
        output.write(" ")
    output.write(letter)
    prev_x = x
    prev_y = y

output.close()