import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import resize
import get_images
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image', help="Enter image path")
arg = parser.parse_args()


img = plt.imread(arg)

img = resize(img, (28,28), anti_aliasing=False)

img = rgb2gray(img)

img = 1-img

img = img/255

plt.imshow(img, cmap="Greys_r")

similar = get_images.get_similar_images(img)

print(similar)