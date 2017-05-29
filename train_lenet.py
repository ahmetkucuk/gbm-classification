import lenet
import tensorflow as tf
from dataset import get_datasets
import sys


def train(args):
	if len(args) < 5:
		print("arguments not valid: 0-> dataset_dir, 1->log_dir, 2-> learning_rate, 3-> image_size, 4-> batch_size, 5-> epoches")
		exit()

	dataset_dir = args[0]
	log_dir = args[1]
	learning_rate = float(args[2])
	image_size = int(args[3])
	batch_size = int(args[4])
	epoches = int(args[5])

	optimizer = tf.train.AdamOptimizer(learning_rate)
	tf.logging.set_verbosity(tf.logging.INFO)

	network_fn = lenet.lenet
	images_placeholder = tf.placeholder("float", [None, image_size, image_size, 1])

	labels_placeholder = tf.placeholder("float", [None, 2])
	logits, end_points = network_fn(images_placeholder, num_classes=2, is_training=True)
	predictions = tf.nn.softmax(logits)
	cross_entropy = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(labels=labels_placeholder, logits=logits))
	with tf.name_scope("evaluations"):
		tf.summary.scalar('loss', cross_entropy)

	train_step = optimizer.minimize(cross_entropy)
	accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(labels_placeholder, 1), tf.argmax(logits, 1)), tf.float32))
	with tf.name_scope("evaluations"):
		tf.summary.scalar('accuracy', accuracy)

	with tf.Session() as sess:
		merged = tf.summary.merge_all()
		train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
		test_writer = tf.summary.FileWriter(log_dir + '/test', sess.graph)

		sess.run(tf.global_variables_initializer())

		saver = tf.train.Saver()

		dataset, validation_dataset = get_datasets(dataset_dir=dataset_dir, image_size=image_size)
		n_of_patches = dataset.size()
		iter = 1
		epoch_count = 1
		print("Number of Patches: " + str(n_of_patches))
		print("Number of Validation Patches: " + str(validation_dataset.size()))
		while (iter * batch_size) / n_of_patches < epoches:

			images, labels, batch_index = dataset.next_batch(batch_size)
			_, classification_probs = sess.run([train_step, predictions], feed_dict={images_placeholder: images, labels_placeholder: labels})
			dataset.set_preds(index=batch_index, batch_size=batch_size, predictions=classification_probs)

			if (iter * batch_size) / n_of_patches > epoch_count:
				dataset.build_heat_map(log_dir)
				epoch_count = (iter * batch_size) / n_of_patches
				summary, ce, acc = sess.run([merged, cross_entropy, accuracy], feed_dict={images_placeholder: images, labels_placeholder: labels})
				train_writer.add_summary(summary=summary, global_step=iter)
				print("Epoches: " + str(epoch_count) + " training loss: " + str(ce))
				print("Epoches: " + str(epoch_count) + " training acc: " + str(acc))

				total_val = 0
				val_iterations = int(validation_dataset.size() / batch_size)
				print("Number of Patches: " + str(val_iterations))
				for i in range(val_iterations):
					val_images, val_labels, _ = validation_dataset.next_batch(batch_size)
					summary, ce, acc = sess.run([merged, cross_entropy, accuracy], feed_dict={images_placeholder: val_images, labels_placeholder: val_labels})
					total_val = total_val + acc
					test_writer.add_summary(summary=summary, global_step=iter)
				print("Epoches: " + str(epoch_count) + " Val accuracy: " + str(float(total_val / val_iterations)))

				if epoch_count % 100 == 0:
					save_path = saver.save(sess, log_dir + "model.ckpt", global_step=iter)
					print("Model saved in file: %s" % save_path)
			if iter % 100 == 0:
				print('At Iteration: ' + str(iter))
			iter = iter + 1
if __name__ == '__main__':
	train(sys.argv[1:])

# args = ["/Users/ahmetkucuk/Documents/Research/Medical/patches/", "/Users/ahmetkucuk/Documents/log_gbm/", 0.01, 32, 50, 3]
# train(args)