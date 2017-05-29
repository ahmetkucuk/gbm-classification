import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


class MnistGanModelTrain(object):

	def __init__(self, model, dataset, name="in_MnistGanModelTrain_Not_Specified"):

		self.D_loss, self.G_loss, self.Q_loss = model.get_losses()
		self.D_solver, self.G_solver, self.Q_solver = model.get_solvers()
		self.G_sample = model.get_g_sample()
		self.X, self.Z, self.c = model.get_placeholders()
		self.dataset = dataset
		self.name = name

	def sample_Z(self, m, n):
		return np.random.uniform(-1., 1., size=[m, n])

	def sample_c(self, m):
		return np.random.multinomial(1, 10*[0.1], size=m)

	def plot(self, samples):
		fig = plt.figure(figsize=(4, 4))
		gs = gridspec.GridSpec(4, 4)
		gs.update(wspace=0.05, hspace=0.05)

		for i, sample in enumerate(samples):
			ax = plt.subplot(gs[i])
			plt.axis('off')
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.set_aspect('equal')
			plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

		return fig

	def log_print(self, val_str):
		print("Log in " + self.name + ":\t" + val_str)

	def train(self, n_of_epochs, n_of_samples, should_plot=False, batch_size=50):

		mb_size = batch_size
		Z_dim = 16

		sess = tf.Session()
		sess.run(tf.global_variables_initializer())

		if not os.path.exists('out/'):
			os.makedirs('out/')

		i = 0

		n_of_iter = int(((self.dataset.size()*n_of_epochs)/batch_size))
		self.log_print("Number of iteration %d" % n_of_iter)
		for it in range(n_of_iter):
			if it % 10000 == 0:
				generated_samples = []
				generated_labels = []
				for k in range(n_of_samples/Z_dim):
					Z_noise = self.sample_Z(16, Z_dim)

					idx = np.random.randint(0, 10)
					c_noise = np.zeros([16, 10])
					c_noise[range(16), idx] = 1
					generated_labels.append(c_noise)

					samples = sess.run(self.G_sample, feed_dict={self.Z: Z_noise, self.c: c_noise})
					generated_samples.append(samples)

				for index in range(10):
					samples = generated_samples[index]
					fig = self.plot(samples)
					plt.savefig('out/{}_{}_{}.png'.format(self.name, str(index).zfill(3), it), bbox_inches='tight')
					i += 1
					plt.close(fig)
				#return np.concatenate(generated_samples), np.concatenate(generated_labels)

			X_mb, _ = self.dataset.next_batch(mb_size)
			X_mb = np.reshape(X_mb, [50, 784])

			Z_noise = self.sample_Z(mb_size, Z_dim)
			c_noise = self.sample_c(mb_size)

			_, D_loss_curr = sess.run([self.D_solver, self.D_loss],
									  feed_dict={self.X: X_mb, self.Z: Z_noise, self.c: c_noise})

			_, G_loss_curr = sess.run([self.G_solver, self.G_loss],
									  feed_dict={self.Z: Z_noise, self.c: c_noise})

			sess.run([self.Q_solver], feed_dict={self.Z: Z_noise, self.c: c_noise})

			if it % 1000 == 0:
				self.log_print('Iter: {}'.format(it))
				self.log_print('D loss: {:.4}'. format(D_loss_curr))
				self.log_print('G_loss: {:.4}'.format(G_loss_curr))