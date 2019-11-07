# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
from PIL import Image
import math

img_size = 56
BUFFER_SIZE = 202599
BATCH_SIZE = 256

data_dir = 'D:/研究所/碩一/基於深度學習之視覺辯論專論/HW2/HW2/data'

def read_and_decode(filename):
  filename_queue = tf.train.string_input_producer([data_dir + ".tfrecords"], num_epochs = None)
    
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(serialized_example, features = {'image_raw':tf.FixedLenFeature([], tf.string)})
  image = tf.decode_raw(features['image_raw'], tf.uint8)

  image = tf.reshape(image, [BUFFER_SIZE, img_size, img_size, 3])
  image = tf.cast(image, tf.float32)
  image = (image - 127.5) / 127.5 # Normalize the images to [-1, 1]
  print("Decode DONE!")

  return image

image_list = read_and_decode(data_dir + ".tfrecords")
print(type(image_list))
print(image_list.shape)

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(image_list).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
iterator = train_dataset.make_initializable_iterator()
next_element = iterator.get_next()
limit = tf.placeholder(dtype = tf.int32, shape = [])

print(type(train_dataset))
print(train_dataset)

def make_generator_model():
    model = tf.keras.Sequential()
    # Input = (100,), Output = (7*7*256*3,) = (12544*3,)
    model.add(layers.Dense(7*7*256*3, use_bias = False, input_shape = (100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Input = (12544*3,), Output = (7, 7, 768)
    model.add(layers.Reshape((7, 7, 768)))
    assert model.output_shape == (None, 7, 7, 768) # Note: None is the batch size

    # Input = (7, 7, 768), Output = (7, 7, 384)
    model.add(layers.Conv2DTranspose(384, (5, 5), strides = (1, 1), padding = 'same', use_bias = False))
    assert model.output_shape == (None, 7, 7, 384)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Input = (7, 7, 384), Output = (14, 14, 192)
    model.add(layers.Conv2DTranspose(192, (5, 5), strides = (2, 2), padding = 'same', use_bias = False))
    assert model.output_shape == (None, 14, 14, 192)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Input = (14, 14, 192), Output = (28, 28, 96)
    model.add(layers.Conv2DTranspose(96, (5, 5), strides = (2, 2), padding = 'same', use_bias = False))
    assert model.output_shape == (None, 28, 28, 96)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Input = (28, 28, 96), Output = (56, 56, 3)
    model.add(layers.Conv2DTranspose(3, (5, 5), strides = (2, 2), padding = 'same', use_bias = False, activation = 'tanh'))
    assert model.output_shape == (None, 56, 56, 3)

    return model

generator = make_generator_model()

noise = tf.random_normal([1, 100])
generated_image = generator(noise, training = False)

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides = (2, 2), padding = 'same',
                                     input_shape = [56, 56, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides = (2, 2), padding = 'same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

discriminator = make_discriminator_model()
decision = discriminator(generated_image)
def discriminator_loss(true_label, fake_label, real_output, fake_output):
    real_loss = tf.keras.losses.binary_crossentropy()(true_label, real_output)
    fake_loss = tf.keras.losses.binary_crossentropy()(fake_label, fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(true_label, fake_output):
    return tf.keras.losses.binary_crossentropy()(true_label, fake_output)

generator_optimizer = tf.train.AdamOptimizer(0.0002, 0.5)
discriminator_optimizer = tf.train.AdamOptimizer(0.0002, 0.5)

checkpoint_dir = './training_checkpoints3'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer = generator_optimizer,
                                 discriminator_optimizer = discriminator_optimizer,
                                 generator = generator,
                                 discriminator = discriminator)

"""## Define the training loop"""

EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 9

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random_normal([num_examples_to_generate, noise_dim])

"""The training loop begins with generator receiving a random seed as input. That seed is used to produce an image. The discriminator is then used to classify real images (drawn from the training set) and fakes images (produced by the generator). The loss is calculated for each of these models, and the gradients are used to update the generator and discriminator."""

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      # Train Discriminator
      generated_images = generator(noise, training = False)

      real_output = discriminator(images, training = True)
      fake_output = discriminator(generated_images, training = True)

      disc_true_label = tf.fill(real_output.shape, 0.9)
      disc_fake_label = tf.zeros_like(fake_output)
      disc_loss = discriminator_loss(disc_true_label, disc_fake_label, real_output, fake_output)

      # Train Generator
      generated_images = generator(noise, training = True)
      fake_output = discriminator(generated_images, training = False)
      gen_true_label = tf.fill(fake_output.shape, 0.9)
      gen_loss = generator_loss(gen_true_label, fake_output)
    
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    gen_losses = []
    disc_losses = []
    
    with tf.Session() as sess:
      sess.run(iterator.initializer, feed_dict={limit: 10})
      for i in range(BUFFER_SIZE//BATCH_SIZE):
        image_batch = sess.run(next_element)
        gen_loss, disc_loss = train_step(image_batch)
        gen_losses.append(gen_loss)
        disc_losses.append(disc_loss)


    # Produce images for the GIF as we go
    display.clear_output(wait = True)
    generate_and_save_images(generator, epoch + 1, seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 5 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    mean_gen_losses = tf.reduce_mean(gen_losses)
    mean_disc_losses = tf.reduce_mean(disc_losses)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
    print('Generator loss:{}'.format(mean_gen_losses))
    print('Discriminator loss:{}'.format(mean_disc_losses))

  # Generate after the final epoch
  display.clear_output(wait = True)
  generate_and_save_images(generator, epochs, seed)

"""**Generate and save images**"""

def output_fig(images_array, file_name):
    plt.figure(figsize = (6, 6), dpi = 100)
    # Get maximum size for square grid of images
    save_size = math.floor(np.sqrt(images_array.shape[0]))

    # Scale to 0-255
    images_array = tf.cast(images_array * 127.5 + 127.5, dtype = tf.uint8)

    # Put images in a square arrangement
    images_in_square = np.reshape(
            images_array[:save_size*save_size],
            (save_size, save_size, images_array.shape[1], images_array.shape[2], images_array.shape[3]))

    # Combine images to grid image
    new_im = Image.new('RGB', (images_array.shape[1] * save_size, images_array.shape[2] * save_size))
    for col_i, col_images in enumerate(images_in_square):
        for image_i, image in enumerate(col_images):
            im = Image.fromarray(image, 'RGB')
            new_im.paste(im, (col_i * images_array.shape[1], image_i * images_array.shape[2]))
    plt.imshow(new_im)
    plt.axis("off")
    plt.savefig(file_name+'.png', bbox_inches='tight', pad_inches=0)    
    #plt.show()

#@title Default title text
def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training = False)
  output_fig(predictions, "./result/results_1_{:02d}".format(epoch))

"""## Train the model
Call the `train()` method defined above to train the generator and discriminator simultaneously. Note, training GANs can be tricky. It's important that the generator and discriminator do not overpower each other (e.g., that they train at a similar rate).

At the beginning of the training, the generated images look like random noise. As training progresses, the generated digits will look increasingly real. After about 50 epochs, they resemble MNIST digits. This may take about one minute / epoch with the default settings on Colab.

Restore the latest checkpoint.
"""

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

train(train_dataset, EPOCHS)

checkpoint.save(file_prefix = checkpoint_prefix)

for i in range(500):
    generated_im = tf.random.normal([num_examples_to_generate, noise_dim])
    result = generator(generated_im, training = False)
    # print(result.shape) # should be (9, width, height, 3)
    output_fig(result, file_name = "./images/{}_image".format(str.zfill(str(i), 3)))