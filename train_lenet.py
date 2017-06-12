import lenet
import tensorflow as tf
from dataset import get_datasets
import sys
from tf_data_pipeline import DataPipeline
from patch_prediction_holder import SlidesPredictionHolder
from tf_data_pipeline import EventFileListTracker
import tensorflow.contrib.slim as slim
import os


def train_by_epochs(data_pipeline, image_size, log_dir, n_of_classes, learning_rate, epochs, batch_size, global_iter):

	with tf.Session() as sess:

		optimizer = tf.train.AdamOptimizer(learning_rate)
		tf.logging.set_verbosity(tf.logging.INFO)

		network_fn = lenet.lenet
		batched_image, batched_labels, batched_file_ids, batched_rows, batched_cols = data_pipeline.get_data()

		batched_labels = slim.one_hot_encoding(labels=batched_labels, num_classes=n_of_classes)
		batched_image = tf.image.resize_images(batched_image, size=[image_size, image_size])
		logits, end_points = network_fn(batched_image, num_classes=n_of_classes, is_training=True)
		predictions = tf.nn.softmax(logits)
		cross_entropy = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits(labels=batched_labels, logits=logits))

		with tf.name_scope("evaluations"):
			tf.summary.scalar('loss', cross_entropy)

		train_step = optimizer.minimize(cross_entropy)
		accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(batched_labels, 1), tf.argmax(logits, 1)), tf.float32))

		with tf.name_scope("evaluations"):
			tf.summary.scalar('accuracy', accuracy)

		n_of_patches = data_pipeline.get_patch_count()

		print("Number of Patches: " + str(n_of_patches))
		slides_predictions = SlidesPredictionHolder()

		merged = tf.summary.merge_all()
		train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
		# test_writer = tf.summary.FileWriter(log_dir + '/test', sess.graph)

		saver = tf.train.Saver()
		is_restored = False

		if tf.gfile.IsDirectory(log_dir):
			checkpoint = tf.train.latest_checkpoint(log_dir)
			if checkpoint is not None:
				print("checkpoint found: " + checkpoint)
				saver.restore(sess, checkpoint)
				is_restored = True
		if not is_restored:
			sess.run(tf.global_variables_initializer())

		epoch_count = 1
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		iter = 1
		while (iter * batch_size) / n_of_patches < epochs:

			if coord.should_stop():
				break

			_, classification_probs, labels, file_ids, rows, cols = sess.run([train_step, predictions, batched_labels, batched_file_ids, batched_rows, batched_cols])

			slides_predictions.set_predictions(classification_probs, labels, file_ids, rows, cols)

			if (iter * batch_size) / n_of_patches > epoch_count:

				epoch_count = (iter * batch_size) / n_of_patches

				summary, ce, acc = sess.run([merged, cross_entropy, accuracy])
				train_writer.add_summary(summary=summary, global_step=global_iter)
				print("Epoches: " + str(epoch_count) + " training loss: " + str(ce))
				print("Epoches: " + str(epoch_count) + " training acc: " + str(acc))

			iter = iter + 1

		save_path = saver.save(sess, log_dir + "model.ckpt", global_step=global_iter)
		print("Model saved in file: %s" % save_path)
		image_output_dir = os.path.join(log_dir, "images")
		if not os.path.exists(image_output_dir):
			os.mkdir(image_output_dir)
		slides_predictions.build_heat_map(output_dir=image_output_dir, epochs=global_iter)
		coord.request_stop()
		coord.join(threads)
		return iter, slides_predictions


def train(args):
	if len(args) < 5:
		print("arguments not valid: 0-> dataset_dir, 1->log_dir, 2-> learning_rate, 3-> image_size, 4-> batch_size, 5-> epoches")
		exit()

	dataset_dir = args[0]
	log_dir = args[1]
	learning_rate = float(args[2])
	image_size = int(args[3])
	batch_size = int(args[4])
	epochs = int(args[5])
	event_file_list_tracker = EventFileListTracker(dataset_dir)
	n_of_classes = 8
	iterations = 1
	while epochs >= 0:
		with tf.Graph().as_default():
			data_pipeline = DataPipeline(event_file_list_tracker=event_file_list_tracker, batch_size=batch_size)
			iter_count, slides_predictions = train_by_epochs(data_pipeline=data_pipeline, image_size=image_size, log_dir=log_dir, n_of_classes=n_of_classes, learning_rate=learning_rate, epochs=2, batch_size=batch_size, global_iter=iterations)
			event_file_list_tracker.filter_out_low_probs(slides_predictions)
			iterations += iter_count
			epochs -= 2
if __name__ == '__main__':
	train(sys.argv[1:])

# args = ["/Users/ahmetkucuk/Documents/Research/Medical/patches/", "/Users/ahmetkucuk/Documents/log_test/", 0.01, 32, 20, 20]
# train(args)