from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras import objectives
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from scipy.stats import norm

import matplotlib.pyplot as plt

import numpy as np

path_save_root = "/blue/cis6930/d.bodra/ML/VAE/"

batch_size, n_epoch = 100, 200
n_hidden, z_dim = 256, 2

n = 15
digit_size = 28

(train_x, trainm_y), (test_x, testm_y) = mnist.load_data()
train_x, test_x = train_x.astype('float32')/255.0, test_x.astype('float32')/2550.
train_x, test_x = train_x.reshape(train_x.shape[0], -1), test_x.reshape(test_x.shape[0], -1)

encoder = Input(shape=(train_x.shape[1:]))
encoder = Dense(n_hidden, activation='relu')(encoder)
encoder = Dense(n_hidden//2, activation='relu')(encoder)
mean = Dense(z_dim)(encoder)
log_var = Dense(z_dim)(encoder)

def sample(args):
  mean, log_var = args
  eps = K.random_normal(shape=(batch_size, z_dim), mean=0., stddev=1.0)
  return mean + K.exp(log_var) * eps

z = Lambda(sample, output_shape=(z_dim,))([mean, log_var])

decoder_1 = Dense(n_hidden//2, activation='relu')
decoder_2 = Dense(n_hidden, activation='relu')
decoder = Dense(train_x.shape[1], activation='sigmoid')

decoded = decoder_1(z)
decoded = decoder_2(decoded)
y = decoder(decoded)

reconstruction_loss = objectives.binary_crossentropy(x, y) * train_x.shape[1]
kl_loss = 0.5 * K.sum(K.square(mean) + K.exp(log_var) - log_var - 1, axis = -1)
vae_loss = reconstruction_loss + kl_loss

vae = Model(x, y)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()

vae.fit(train_x,
  shuffle=True,
  epochs=n_epoch,
  batch_size=batch_size,
  validation_data=(test_x, None), verbose=1)

encoder = Model(x, mean)
encoder.summary()

decoder_input = Input(shape=(z_dim,))
m_decoded = decoder_1(decoder_input)
m_decoded = decoder_2(m_decoded)
m_y = decoder(m_decoded)
generator = Model(decoder_input, m_y)
generator.summary()

filename = path_save_root+'models/vae_gen_model.h5'
generator.save(filename)


figure = np.zeros((digit_size * n, digit_size * n))

grid_x = norm.ppf(np.linspace(0.05, 0.95, n)) 
gridm_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(gridm_y):
        z_sample = np.array([[xi, yi]])
        x_m_decoded = generator.predict(z_sample)
        digit = x_m_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='gray_r')
filename = path_save_root+'results/'+'vae_plot_e%03d.png' % (n_epoch)
plt.savefig(filename)
plt.close()