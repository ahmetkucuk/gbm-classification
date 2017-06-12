import os
import tensorflow as tf
from scipy import misc
import numpy as np
from PIL import Image

from sklearn.utils import shuffle


dataset_dir = "/Users/ahmetkucuk/Documents/Research/Medical/patches/"


def get_label_from_filename(filename):

	if "/AII/" in filename:
		return 0
	elif "/AIII/" in filename:
		return 1
	elif "/OII/" in filename:
		return 2
	elif "/OIII/" in filename:
		return 3
	elif "/OAII/" in filename:
		return 4
	elif "/OAIII/" in filename:
		return 5
	elif "/GBM/" in filename:
		return 6
	else:
		return 7


def get_metadata(filename):

	metadata = []
	metadata.append(get_label_from_filename(filename))
	filename = filename.split("/")[-1]
	metadata_tuple = filename.split("_")

	metadata.append(metadata_tuple[0])
	image_segment = metadata_tuple[3]
	index_i = image_segment.index("i")
	index_j = image_segment.index("j")
	index_dot = image_segment.index(".jpg")

	metadata.append(int(image_segment[index_i+1:index_j]))
	metadata.append(int(image_segment[index_j+1:index_dot]))
	return metadata


def create_label(filepath):
	return get_metadata(filepath)


def get_metadata_from_list(filenames):
	metadatas = []
	for f in filenames:
		metadatas.append(get_metadata(f))
	return metadatas


def _get_filenames_and_classes(dataset_dir):
	directories = []
	class_names = []
	for filename in os.listdir(dataset_dir):
		path = os.path.join(dataset_dir, filename)
		if os.path.isdir(path):
			directories.append(path)
			class_names.append(filename)

	photo_filenames = []
	for directory in directories:
		for filename in os.listdir(directory):
			if ".jpg" in filename:
				path = os.path.join(directory, filename)
				photo_filenames.append(path)

	return photo_filenames, sorted(class_names)


class EventFileListTracker(object):

	def __init__(self, dataset_dir):
		self.filenames, classes = _get_filenames_and_classes(dataset_dir=dataset_dir)

		metadatas = get_metadata_from_list(self.filenames)
		self.metadata_labels = [i[0] for i in metadatas]
		self.metadata_file_ids = [i[1] for i in metadatas]
		self.metadata_rows = [i[2] for i in metadatas]
		self.metadata_cols = [i[3] for i in metadatas]
		print(self.filenames)
		print(self.metadata_labels)

	def get_data(self):
		return self.filenames, self.metadata_labels, self.metadata_file_ids, self.metadata_rows, self.metadata_cols

	def size(self):
		return len(self.filenames)

	def filter_out_low_probs(self, slides_predictions):
		slide_prediction_holders_by_id = slides_predictions.get_predictions()

		predictions = []
		for slide in slide_prediction_holders_by_id.keys():
			predictions_by_patch_position = slide_prediction_holders_by_id[slide]
			for position in predictions_by_patch_position.keys():
				predictions.append([slide, position, predictions_by_patch_position[position]])
		print(predictions)
		predictions.sort(key=lambda x: max(x[2]))
		toberemoved = predictions[0:int(len(predictions)*0.05)]

		new_filenames = []
		new_metadata_labels = []
		new_metadata_file_ids = []
		new_metadata_rows = []
		new_metadata_cols = []

		for f, l, id, r, c in zip(self.filenames, self.metadata_labels, self.metadata_file_ids, self.metadata_rows, self.metadata_cols):
			shouldAdd = True
			for remove in toberemoved:
				if remove[0] == id and remove[1].row == r and remove[1].col == c:
					shouldAdd = False
			if shouldAdd:
				new_filenames.append(f)
				new_metadata_labels.append(l)
				new_metadata_file_ids.append(id)
				new_metadata_rows.append(r)
				new_metadata_cols.append(c)

		self.filenames = new_filenames
		self.metadata_labels = new_metadata_labels
		self.metadata_file_ids = new_metadata_file_ids
		self.metadata_rows = new_metadata_rows
		self.metadata_cols = new_metadata_cols


class DataPipeline(object):

	def __init__(self, event_file_list_tracker, batch_size):

		filenames, metadata_labels, metadata_file_ids, metadata_rows, metadata_cols = event_file_list_tracker.get_data()
		self.n_of_patches = len(filenames)
		filenames, metadata_labels, metadata_file_ids, metadata_rows, metadata_cols = shuffle(filenames, metadata_labels, metadata_file_ids, metadata_rows, metadata_cols)

		image_queue = tf.FIFOQueue(capacity=50, dtypes=[tf.string], shapes=[[]])
		metadata_queue = tf.FIFOQueue(capacity=50, dtypes=[tf.int32, tf.string, tf.int32, tf.int32], shapes=[[], [], [], []])

		image_enqueue_op = image_queue.enqueue_many([filenames])
		metadata_enqueue_op = metadata_queue.enqueue_many([metadata_labels, metadata_file_ids, metadata_rows, metadata_cols])
		labels, file_ids, rows, cols = metadata_queue.dequeue()

		reader = tf.WholeFileReader()
		key, value = reader.read(image_queue)

		my_img = tf.image.decode_jpeg(value)
		my_img.set_shape([256, 256, 3])
		my_img = tf.image.per_image_standardization(my_img)

		self.batched_image = tf.train.batch([my_img], batch_size=batch_size)

		self.batched_labels, self.batched_file_ids, self.batched_rows, self.batched_cols = tf.train.batch([labels, file_ids, rows, cols], batch_size=batch_size)

		tf.train.queue_runner.add_queue_runner(
			tf.train.queue_runner.QueueRunner(image_queue, [image_enqueue_op]*5))

		tf.train.queue_runner.add_queue_runner(
			tf.train.queue_runner.QueueRunner(metadata_queue, [metadata_enqueue_op]*5))

	def get_data(self):
		return self.batched_image, self.batched_labels, self.batched_file_ids, self.batched_rows, self.batched_cols

	def get_patch_count(self):
		return self.n_of_patches


def train_for_limited_epochs(input_numbers, epochs, batch_size, iter, checkpoint_path):
	with tf.Graph().as_default(), tf.Session() as sess:

		biases = tf.Variable(tf.zeros([1]), name="biases")

		coord = tf.train.Coordinator()

		queue = tf.train.input_producer(input_tensor=input_numbers, num_epochs=epochs)
		sess.run(tf.local_variables_initializer())

		numbers = queue.dequeue()
		batch_numbers = tf.train.batch([numbers], batch_size=batch_size)
		batch_numbers = tf.multiply(batch_numbers, 10.0) + biases

		saver = tf.train.Saver()
		is_restored = False

		if tf.gfile.IsDirectory(checkpoint_path):
			checkpoint = tf.train.latest_checkpoint(checkpoint_path)
			if checkpoint is not None:
				print("checkpoint found: " + checkpoint)
				saver.restore(sess, checkpoint)
				is_restored = True
		if not is_restored:
			sess.run(tf.global_variables_initializer())

		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		for step in xrange(5):
			if coord.should_stop():
				break
			batch = sess.run([batch_numbers])
			print(batch)
			iter += 1

		save_path = saver.save(sess, checkpoint_path + "model.ckpt", global_step=iter)
		print("Model saved in file: %s" % save_path)

		coord.request_stop()
		coord.join(threads)

# input_length = 1000
# batch_size = 20
# epochs = 2
# iter_count = (input_length/batch_size) * epochs
# iter = 0
# checkpoint_path = "/Users/ahmetkucuk/Documents/log_gbm/"
# for k in range(100):
# 	input_numbers = [k*1.0 for i in range(input_length)]
# 	train_for_limited_epochs(input_numbers, epochs=epochs, batch_size=batch_size, iter=iter, checkpoint_path=checkpoint_path)
# 	iter += iter_count

# from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
#
# checkpoint_path = "/Users/ahmetkucuk/Documents/log_gbm/"
# checkpoint = tf.train.latest_checkpoint(checkpoint_path)
# print_tensors_in_checkpoint_file(file_name=checkpoint, tensor_name='', all_tensors=True)

# with tf.Session() as sess:
#
# 	biases = tf.Variable(tf.zeros([1]), name="biases")
# 	sess.run(tf.global_variables_initializer())
# 	coord = tf.train.Coordinator()
# 	queue = tf.FIFOQueue(capacity=50, dtypes=[tf.float32], shapes=[[]])
# 	numbers = queue.dequeue()
# 	batch_numbers = tf.train.batch([numbers], batch_size=20)
# 	batch_numbers = tf.multiply(batch_numbers, 10.0) + biases # simulate a network operation
# 	#Need to change the content of the queue 10 times
# 	for k in range(10):
#
# 		input_numbers = [k*1.0 for i in range(100)]
#
# 		queue_op = queue.enqueue_many([input_numbers])
#
# 		qr = tf.train.queue_runner.QueueRunner(queue, [queue_op]*5)
# 		tf.train.queue_runner.add_queue_runner(qr)
#
# 		threads = tf.train.start_queue_runners(coord=coord)
#
# 		for step in xrange(100):
# 			if coord.should_stop():
# 				break
# 			batch = sess.run([batch_numbers])
# 			print(batch)
# 		print("finished that batch")
