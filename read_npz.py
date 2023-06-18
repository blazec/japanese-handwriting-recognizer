import numpy as np
from matplotlib import pyplot as plt

def show_img(img):
     plt.imshow(img, cmap=plt.cm.binary)
     plt.show()

if __name__ == '__main__':
    np_arr = np.load('hiragana_images.npz')
    image_array = np_arr['arr_0']
    image = image_array[0,0]
    show_img(image)




