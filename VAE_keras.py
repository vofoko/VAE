import tensorflow as tf
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
import random


def show_image(img):
    plt.imshow(img)
    plt.show()

def random_flip_horizontal(image):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
    return image


def random_transpose(image):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.transpose(image)
        
    return image

def random_rotate(image):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.rot90(image)
    return image


def random_hue(image):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_hue(image, 0.1)
    return image

def random_contrast(image):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_contrast(
            image, lower=0.5, upper=0.9)
    return image

def random_brightness(image):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_brightness(
            image, max_delta=0.2)
    return image

def random_scaled(image):
    if tf.random.uniform(()) > 0.5:
        image = image/255
    return image

def load_image(fname, start = 0, stop = 10):
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img



class TrainDataset(tf.keras.utils.Sequence):
    def __init__(self, fname_list, SIZE, flag_train = True):
        self.flag_train = flag_train
        self.fname_list = fname_list
        random.shuffle(self.fname_list)
                                                                                      
    def __len__(self):
        return len(self.fname_list)

    def __getitem__(self, idx):
        img = load_image(self.fname_list[idx])
        if img.shape[1] != SIZE:
            img = cv2.resize(img, (SIZE, SIZE)).reshape((SIZE, SIZE, 1))
        
        if self.flag_train:
            img = random_flip_horizontal(img)
            img = random_transpose(img)
            img = random_rotate(img)
            img = random_brightness(img)

        return np.array(img).astype(np.float64)/255, np.array(img).astype(np.float64)/255



class VAE:
    def __init__(self, input_shape, z_dim, layers_filter, strides):
        self.input_shape = input_shape
        self.z_dim = z_dim
        self.layers_filter = layers_filter
        self.strides = strides
        self.n_layers_encoder = len(layers_filter)
        self.model = None
        self.encoder = None
        self.decoder = None


    def _build(self):
        encoder_input = tf.keras.layers.Input(self.input_shape)
        x = encoder_input
        for i in range(self.n_layers_encoder):
            x = tf.keras.layers.Conv2D(self.layers_filter[i], 3, strides = self.strides[i], padding = 'same', activation = 'relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
        
        shape_before_flattening = K.int_shape(x)[1:]
        x = tf.keras.layers.Flatten()(x)

        mu = tf.keras.layers.Dense(self.z_dim)(x)
        log_var = tf.keras.layers.Dense(self.z_dim)(x)

        def noiser(args):
            global mu, log_var
            mu, log_var = args
            N = K.random_normal(shape=(K.shape(mu)), mean=0., stddev=1.0)
            return K.exp(log_var / 2) * N + mu

        h = tf.keras.layers.Lambda(noiser, output_shape=(self.z_dim,))([mu, log_var])

        decoder_input = tf.keras.layers.Input(shape=(self.z_dim,))
        x = tf.keras.layers.Dense(np.prod(np.array(shape_before_flattening)))(decoder_input)
        x = tf.keras.layers.Reshape(shape_before_flattening)(x)

        for i in range(self.n_layers_encoder):
            x = tf.keras.layers.Conv2DTranspose(self.layers_filter[-i - 1], 3, strides = self.strides[i], padding = 'same', activation = 'relu')(x)
            x = tf.keras.layers.Dropout(0.25)(x)
            x = tf.keras.layers.BatchNormalization()(x)

        decoded = tf.keras.layers.Conv2D(self.input_shape[-1], 3, activation='sigmoid', padding='same')(x)
        
        self.encoder = tf.keras.Model(encoder_input, h, name='encoder')
        self.decoder = tf.keras.Model(decoder_input, decoded, name='decoder')
        self.model = tf.keras.Model(encoder_input, self.decoder(self.encoder(encoder_input)), name="vae")

    def _compile(self, learning_rate = 0.001, k = 1):
        def vae_loss(x, y):
            loss = K.sum(K.square(x-y))
            kl_loss = -0.5 * K.sum(1 + log_var - K.square(mu) - K.exp(log_var), axis=-1)
            return k*loss + kl_loss

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer= optimizer, loss = vae_loss)
