import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import string

sensitivity = 0.01
letter_size = 0.3

filename = "./test.PNG"
font_path = "./font"

text = np.array(Image.open(filename).convert('L'))
text = 255 - text
plt.imshow(text, cmap='Greys')
plt.show()

letters = list(string.ascii_letters)+list(string.punctuation)[:17]+list(string.digits)

for l in letters:
    f = os.path.join(font_path, l+".png")
    letter = Image.open(f).convert('L')
    letter = letter.resize((int(letter.width*letter_size), int(letter.height*letter_size)))
    letter = np.array(letter)
    letter = 255 - letter


    plt.imshow(letter, cmap='Greys')
    plt.show()

    C = np.real(np.fft.ifft2(np.fft.fft2(text) * np.fft.fft2(np.rot90(letter, 2), text.shape)))

    print(np.amax(C))

    indices = np.nonzero(C >= (1-sensitivity)*np.amax(C))
    x, y = [], []
    for i in range(len(indices[0])):
        y.append(indices[0][i])
        x.append(indices[1][i])


    plt.imshow(text, cmap='Greys')
    plt.plot(x,y, 'ro')

    plt.show()
