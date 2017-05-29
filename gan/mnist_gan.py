import tensorflow as tf

'''
	This is an implementation of infoGAN
	See Reference: https://arxiv.org/abs/1606.03657
'''
class MnistGanModel(object):

	def __init__(self):
		self.init_model()

	def xavier_init(self, size):
		in_dim = size[0]
		xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
		return tf.random_normal(shape=size, stddev=xavier_stddev)

	def init_model(self):

		self.X = tf.placeholder(tf.float32, shape=[None, 784])

		D_W1 = tf.Variable(self.xavier_init([784, 128]))
		D_b1 = tf.Variable(tf.zeros(shape=[128]))

		D_W2 = tf.Variable(self.xavier_init([128, 1]))
		D_b2 = tf.Variable(tf.zeros(shape=[1]))

		theta_D = [D_W1, D_W2, D_b1, D_b2]


		self.Z = tf.placeholder(tf.float32, shape=[None, 16])
		self.c = tf.placeholder(tf.float32, shape=[None, 10])

		G_W1 = tf.Variable(self.xavier_init([26, 256]))
		G_b1 = tf.Variable(tf.zeros(shape=[256]))

		G_W2 = tf.Variable(self.xavier_init([256, 784]))
		G_b2 = tf.Variable(tf.zeros(shape=[784]))

		theta_G = [G_W1, G_W2, G_b1, G_b2]


		Q_W1 = tf.Variable(self.xavier_init([784, 128]))
		Q_b1 = tf.Variable(tf.zeros(shape=[128]))

		Q_W2 = tf.Variable(self.xavier_init([128, 10]))
		Q_b2 = tf.Variable(tf.zeros(shape=[10]))

		theta_Q = [Q_W1, Q_W2, Q_b1, Q_b2]


		self.G_sample = self.generator(G_W1, G_b1, G_W2, G_b2, self.Z, self.c)
		D_real = self.discriminator(D_W1, D_b1, D_W2, D_b2, self.X)
		D_fake = self.discriminator(D_W1, D_b1, D_W2, D_b2, self.G_sample)
		Q_c_given_x = self.Q(Q_W1, Q_b1, Q_W2, Q_b2, self.G_sample)

		self.D_loss = -tf.reduce_mean(tf.log(D_real + 1e-8) + tf.log(1 - D_fake + 1e-8))
		self.G_loss = -tf.reduce_mean(tf.log(D_fake + 1e-8))

		cross_ent = tf.reduce_mean(-tf.reduce_sum(tf.log(Q_c_given_x + 1e-8) * self.c, 1))
		ent = tf.reduce_mean(-tf.reduce_sum(tf.log(self.c + 1e-8) * self.c, 1))
		self.Q_loss = cross_ent + ent

		self.D_solver = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.D_loss, var_list=theta_D)
		self.G_solver = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.G_loss, var_list=theta_G)
		self.Q_solver = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.Q_loss, var_list=theta_G + theta_Q)

	def get_losses(self):
		return self.D_loss, self.G_loss, self.Q_loss

	def get_solvers(self):
		return self.D_solver, self.G_solver, self.Q_solver

	def get_g_sample(self):
		return self.G_sample

	def get_placeholders(self):
		return self.X, self.Z, self.c

	def generator(self, G_W1, G_b1, G_W2, G_b2, z, c):
		inputs = tf.concat([z, c], 1)
		G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
		G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
		G_prob = tf.nn.sigmoid(G_log_prob)

		return G_prob

	def discriminator(self, D_W1, D_b1, D_W2, D_b2, x):
		D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
		D_logit = tf.matmul(D_h1, D_W2) + D_b2
		D_prob = tf.nn.sigmoid(D_logit)

		return D_prob

	def Q(self, Q_W1, Q_b1, Q_W2, Q_b2, x):
		Q_h1 = tf.nn.relu(tf.matmul(x, Q_W1) + Q_b1)
		Q_prob = tf.nn.softmax(tf.matmul(Q_h1, Q_W2) + Q_b2)

		return Q_prob

