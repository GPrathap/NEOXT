import scipy.misc
import numpy as np

def inverse_transform(images):
    return (images+1.)/2.

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img