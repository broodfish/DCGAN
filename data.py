import os
import tensorflow as tf 
from PIL import Image
import numpy as np

data_dir = 'D:/研究所/碩一/基於深度學習之視覺辯論專論/HW2/HW2/data'

img_size = 56

def getTrainList(data_dir):
    images = []
    for dirPath, _, fileNames in os.walk(data_dir):
        for name in fileNames:
            images.append(os.path.join(dirPath, name))
    
    return images

def bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def trans2tfRecord(images, filename):
    n_samples = len(images)
    TFWriter = tf.python_io.TFRecordWriter(filename + '.tfrecords')
    for i in np.arange(0, n_samples):
        try:
            image = Image.open(images[i])
            image = image.resize((img_size, img_size),Image.ANTIALIAS)
            image = np.asarray(image, dtype = np.float32)
            if image is None:
                print('Error image:' + images[i])
            else:
                image_raw = image.tostring()
            
            ftrs = tf.train.Features(
                    feature = {'image_raw': bytes_feature(image_raw),
                               })
            example = tf.train.Example(features = ftrs)
            TFWriter.write(example.SerializeToString())
        except IOError as e:
            print('Skip!\n')
    TFWriter.close()
    print('Transform done!')

images = getTrainList(data_dir)
trans2tfRecord(images, data_dir)