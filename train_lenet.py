import tensorflow as tf
from dataset import get_data
import lenet
from sklearn import preprocessing

dataset_dir = "/home/ahmet/Documents/Research/Medical/Data/bootstrap1500/train"
learning_rate = 0.1
batch_size = 50
image_dim = 28
num_classes = 8

lb = preprocessing.LabelBinarizer()
lb.fit([0, 1, 2, 3, 4, 5, 6, 7])


images, labels = get_data(dataset_dir)
labels = lb.transform(labels)

image_input = tf.placeholder(tf.float32, [batch_size, 256, 256, 3], name='image_input')
image_resized_input = tf.image.resize_images(images=image_input, size=[image_dim, image_dim])
label_input = tf.placeholder(tf.int16, [batch_size, num_classes], name='label_input')
logits, end_points = lenet.lenet(image_resized_input, num_classes=num_classes)

cross_entropy = tf.reduce_mean(
	tf.nn.softmax_cross_entropy_with_logits(labels=label_input, logits=logits))

cross_entropy = tf.Print(cross_entropy, [cross_entropy], name='cross_entropy')


optimizer = tf.train.GradientDescentOptimizer(cross_entropy)
# Create a variable to track the global step.
global_step = tf.Variable(0, name='global_step', trainable=False)
# Use the optimizer to apply the gradients that minimize the loss
# (and also increment the global step counter) as a single training step.
train_step = optimizer.minimize(cross_entropy, global_step=global_step)


sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# Train
for i in range(len(images)/batch_size):
	batch_xs = images[i*batch_size:i*batch_size+batch_size]
	batch_ys = labels[i*batch_size:i*batch_size+batch_size]
	sess.run(train_step, feed_dict={image_input: batch_xs, label_input: batch_ys})
