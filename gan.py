from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, Reshape, Flatten, Dropout, BatchNormalization, LeakyReLU

from matplotlib import pyplot

import numpy as np

path_save_root = "/blue/cis6930/d.bodra/ML/GAN/"

def get_real_data():
	(trainX, _), (_, _) = load_data()
	X = np.expand_dims(trainX, axis=-1)
	X = X.astype('float32')
	X = 2.0*(X / 255.0)-1.0
	return X

def get_real_samples(dataset, n_samples):
	indexes = np.random.randint(0, dataset.shape[0], n_samples)
	X = dataset[indexes]
	y = np.ones((n_samples, 1))
	return X, y

def get_latent_point(latent_dimensionension, n_samples):
	x_input = np.random.randn(latent_dimensionension * n_samples)
	x_input = x_input.reshape(n_samples, latent_dimensionension)
	return x_input

def get_fake_samples(generator, latent_dimension, n_samples):
	x_input = get_latent_point(latent_dimension, n_samples)
	X = generator.predict(x_input)
	y = np.zeros((n_samples, 1))
	return X, y

def plot_and_save(examples, epoch, n=10):
	for i in range(n * n):
		pyplot.subplot(n, n, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
	filename = path_save_root+'results/'+'generated_plot_e%03d.png' % (epoch+1)
	pyplot.savefig(filename)
	pyplot.close()

def print_accuracy(epoch, generator, discriminator, dataset, latent_dimensionension, n_samples):
	X_real, y_real = get_real_samples(dataset, n_samples)
	_, accuracy_real = discriminator.evaluate(X_real, y_real, verbose=0)
	x_fake, y_fake = get_fake_samples(generator, latent_dimensionension, n_samples)
	_, accuracy_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
	print('ACCURACY REAL: %.0f%% | FAKE: %.0f%%' % (accuracy_real*100, accuracy_fake*100))
	plot_and_save(x_fake, epoch)
	filename = path_save_root+'models/'+'model_%03d.h5' % (epoch + 1)
	generator.save(filename)

class SelfAttention(Layer):
    def __init__(self, channels, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.channels = channels
        self.kernels_f_g = self.channels // 8
        self.kernels_h = self.channels

    def build(self, input_shape):
        kernel_shape_f_g = (1, 1) + (self.channels, self.kernels_f_g)
        kernel_shape_h = (1, 1) + (self.channels, self.kernels_h)

        self.gamma = self.add_weight(name='gamma', shape=[1], initializer='zeros', trainable=True)
        self.kernel_f = self.add_weight(shape=kernel_shape_f_g, initializer='glorot_uniform')
        self.kernel_g = self.add_weight(shape=kernel_shape_f_g, initializer='glorot_uniform')
        self.kernel_h = self.add_weight(shape=kernel_shape_h, initializer='glorot_uniform')
        self.bias_f = self.add_weight(shape=(self.kernels_f_g,), initializer='zeros')
        self.bias_g = self.add_weight(shape=(self.kernels_f_g,), initializer='zeros')
        self.bias_h = self.add_weight(shape=(self.kernels_h,), initializer='zeros')
        super(SelfAttention, self).build(input_shape)

        self.input_spec = InputSpec(ndim=4, axes={3: input_shape[-1]})
        self.built = True

    def call(self, x):
        def make_flat(x):
            return K.reshape(x, shape=[K.shape(x)[0], K.shape(x)[1]*K.shape(x)[2], K.shape(x)[-1]])

        f = K.conv2d(x, kernel=self.kernel_f, strides=(1, 1), padding='same')
        f = K.bias_add(f, self.bias_f)
        g = K.conv2d(x, kernel=self.kernel_g, strides=(1, 1), padding='same')
        g = K.bias_add(g, self.bias_g)
        h = K.conv2d(x, kernel=self.kernel_h, strides=(1, 1), padding='same')
        h = K.bias_add(h, self.bias_h)
        s = tf.matmul(make_flat(g), make_flat(f), transpose_b=True)
        beta = K.softmax(s, axis=-1)
        o = K.batch_dot(beta, make_flat(h))
        o = K.reshape(o, shape=K.shape(x))
        x = self.gamma * o + x
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

def get_generator(latent_dimensionension):
	model = Sequential()
	n_nodes = 128 * 7 * 7
	model.add(Dense(n_nodes, input_dim=latent_dimensionension))
	model.add(BatchNormalization(momentum=0.8))
	model.add(LeakyReLU(alpha=0.2))
	model = SelfAttention(n_nodes)(model)
	model.add(Reshape((7, 7, 128)))
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(BatchNormalization(momentum=0.8))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(BatchNormalization(momentum=0.8))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(1, (7,7), activation='tanh', padding='same'))
	return model

def get_discriminator(input_shape):
	model = Sequential()
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=input_shape))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.2))
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002), metrics=['accuracy'])
	return model

def get_gan(generator, discriminator):
	discriminator.trainable = False
	model = Sequential()
	model.add(generator)
	model.add(discriminator)
	model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002))
	return model

def train(generator, discriminator, gan_model, dataset, latent_dimension, n_epochs=100, n_batch=256):
	bat_per_epo = int(dataset.shape[0] / n_batch)
	half_batch = int(n_batch / 2)
	for i in range(n_epochs):
		for j in range(bat_per_epo):
			X_real, y_real = get_real_samples(dataset, half_batch)
			X_fake, y_fake = get_fake_samples(generator, latent_dimension, half_batch)
			X, y = np.vstack((X_real, X_fake)), np.vstack((y_real, y_fake))
			d_loss, _ = discriminator.train_on_batch(X, y)
			X_gan = get_latent_point(latent_dimension, n_batch)
			y_gan = ones((n_batch, 1))
			g_loss = gan_model.train_on_batch(X_gan, y_gan)
			print('%d, %d/%d,LOSS: discriminator=%.3f, generator=%.3f' % (i+1, j+1, bat_per_epo, d_loss, g_loss))
		if (i+1) % 10 == 0:
			print_accuracy(i, generator, discriminator, dataset, latent_dimension, 100)

latent_dimension = 100
discriminator = get_discriminator((28,28,1))
generator = get_generator(latent_dimension)
gan_model = get_gan(generator, discriminator)
dataset = get_real_data()
train(generator, discriminator, gan_model, dataset, latent_dimension, 200, 256)