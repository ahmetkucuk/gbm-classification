import os
import tensorflow as tf
from scipy import misc
import numpy as np
from PIL import Image

from sklearn.utils import shuffle
from cifarnet_preprocessing import preprocess_image


dataset_dir = "/Users/ahmetkucuk/Documents/Research/Medical/patches/"


# def get_label_from_filename(filename):
#
# 	if "/AII/" in filename:
# 		return 0
# 	elif "/AIII/" in filename:
# 		return 1
# 	elif "/OII/" in filename:
# 		return 2
# 	elif "/OIII/" in filename:
# 		return 3
# 	elif "/OAII/" in filename:
# 		return 4
# 	elif "/OAIII/" in filename:
# 		return 5
# 	elif "/GBM/" in filename:
# 		return 6
# 	else:
# 		return 7

MUT = ["TCGA-CS-5393","TCGA-DB-5270","TCGA-DB-5273","TCGA-DB-5275","TCGA-DB-5276","TCGA-DB-5277","TCGA-DB-A4XB","TCGA-DB-A64X","TCGA-DH-5142","TCGA-DH-5143","TCGA-DH-A66B","TCGA-DH-A66D","TCGA-DU-5851","TCGA-DU-5855","TCGA-DU-6396","TCGA-DU-6542","TCGA-DU-7019","TCGA-DU-8163","TCGA-E1-A7YV","TCGA-FG-7636","TCGA-FG-8185","TCGA-FG-A6J3","TCGA-FG-A87N","TCGA-HT-7475","TCGA-HT-747"]
WT = ["TCGA-DH-5140","TCGA-DU-5847","TCGA-DU-5852","TCGA-DU-5854","TCGA-DU-6402","TCGA-DU-6403","TCGA-DU-6405","TCGA-DU-6406","TCGA-DU-7006","TCGA-DU-7012","TCGA-DU-7013","TCGA-DU-8158","TCGA-DU-8161","TCGA-DU-8162","TCGA-DU-8165","TCGA-FG-A4MU","TCGA-FG-A4MW","TCGA-HT-7469","TCGA-HT-7860","TCGA-HT-8011","TCGA-HT-8019","TCGA-HT-8104","TCGA-HT-8110","TCGA-HT-8564","TCGA-HT-A4DS"]


def get_label_from_filename(filename):
	print(filename)
	if filename[:-11] in MUT:
		print("in MUT")
		return 0
	else:
		print("in WT")
		return 1


def get_metadata(filename):

	metadata = []
	filename = filename.split("/")[-1]
	metadata_tuple = filename.split("_")
	metadata.append(get_label_from_filename(metadata_tuple[0]))

	metadata.append(metadata_tuple[0])
	image_segment = metadata_tuple[1]
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

	photo_filenames = []
	for filename in os.listdir(dataset_dir):
		if ".jpg" in filename:
			path = os.path.join(dataset_dir, filename)
			photo_filenames.append(path)

	return photo_filenames


class EventFileListTracker(object):

	def __init__(self, dataset_dir):
		self.filenames = _get_filenames_and_classes(dataset_dir=dataset_dir)

		metadatas = get_metadata_from_list(self.filenames)
		self.metadata_labels = [i[0] for i in metadatas]
		self.metadata_file_ids = [i[1] for i in metadatas]
		self.metadata_rows = [i[2] for i in metadatas]
		self.metadata_cols = [i[3] for i in metadatas]

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
		predictions.sort(key=lambda x: max(x[2]))
		toberemoved = predictions[0:int(len(predictions)*0.01)]

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

		image_queue = tf.FIFOQueue(capacity=500, dtypes=[tf.string], shapes=[[]])
		metadata_queue = tf.FIFOQueue(capacity=500, dtypes=[tf.int32, tf.string, tf.int32, tf.int32], shapes=[[], [], [], []])

		image_enqueue_op = image_queue.enqueue_many([filenames])
		metadata_enqueue_op = metadata_queue.enqueue_many([metadata_labels, metadata_file_ids, metadata_rows, metadata_cols])
		labels, file_ids, rows, cols = metadata_queue.dequeue()

		reader = tf.WholeFileReader()
		key, value = reader.read(image_queue)

		my_img = tf.image.decode_jpeg(value)
		my_img.set_shape([256, 256, 3])
		#my_img = tf.image.per_image_standardization(my_img)
		my_img = preprocess_image(my_img, 224, 224, is_training=True)

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
			iter += 1

		save_path = saver.save(sess, checkpoint_path + "model.ckpt", global_step=iter)
		print("Model saved in file: %s" % save_path)

		coord.request_stop()
		coord.join(threads)
