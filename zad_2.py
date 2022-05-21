import numpy as np
from PIL import Image
import string

from skimage.feature import peak_local_max

sensitivity = 0.03
letter_size = 0.56

space_len = 50
line_height = 32

filename = "./SourceCodetest.PNG"
font_path = "./SourceCodePro"
output_name = "sans_serif.txt"

text = np.array(Image.open(filename).convert('L'))
text = 255 - text

k = 300
U, E, VT = np.linalg.svd(text)
text = np.zeros((len(U), len(VT)), dtype=float)
for i in range(k):
    text += E[i] * np.outer(U.T[i], VT[i])

letters = list(string.ascii_uppercase)
letters += ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
letter_coords = []

number_dict = {'one': '1', 'two': '2', 'three': '3', 'four': '4',
               'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'}

letter_count = dict()
for l in letters:
    letter_count[l] = 0

for l in letters:
    f = font_path + "/" + l + ".png"
    letter = Image.open(f).convert('L')
    letter = letter.resize((int(letter.width * letter_size), int(letter.height * letter_size)))
    letter = np.array(letter)
    letter = 255 - letter

    C = np.real(np.fft.ifft2(np.fft.fft2(text) * np.fft.fft2(np.rot90(letter, 2), text.shape)))

    coordinates = peak_local_max(C, min_distance=10, threshold_rel=1 - sensitivity)

    for i in range(len(coordinates)):
        letter_count[l] += 1
        if l in number_dict:
            letter_coords.append((coordinates[i][0], coordinates[i][1], number_dict[l]))
        else:
            letter_coords.append((coordinates[i][0], coordinates[i][1], l.lower()))

for i in range(line_height, text.shape[0], line_height):
    for j in range(len(letter_coords)):
        if i > letter_coords[j][0] >= i - line_height:
            letter_coords[j] = (i - line_height, letter_coords[j][1], letter_coords[j][2])

letter_coords.sort()
output = open(output_name, 'w')
prev_y = letter_coords[0][0]
prev_x = letter_coords[0][1]
for y, x, letter in letter_coords:
    if y - prev_y >= line_height:
        output.write("\n")
    elif x - prev_x >= space_len:
        output.write(" ")
    output.write(letter)
    prev_x = x
    prev_y = y

output.close()

print(letter_count)
