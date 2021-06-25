from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from PIL import Image

import os

directory = "path/to/data/dir"

for folder in os.listdir(directory):
    for image in os.listdir(folder):
        img = load_img(image)
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data, 0)
        # create image data augmentation generator
        data_gen = ImageDataGenerator(brightness_range=[0.5, 1.0], rotation_range=25)
        # prepare iterator
        it = data_gen.flow(samples, batch_size=1)
        # generate samples
        for i in range(10):
            batch = it.next()
            im = Image.fromarray(batch[0].astype('uint8'))
            im.save(image + "augmented" + str(i) + ".jpg")
