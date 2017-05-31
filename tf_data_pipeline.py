import os
import tensorflow as tf
from scipy import misc
import numpy as np
from PIL import Image
from dataset import _get_filenames_and_classes

from sklearn.utils import shuffle


dataset_dir = "/Users/ahmetkucuk/Documents/Research/Medical/patches/"


def get_metadata(filename):

	metadata = []
	filename = filename.split("/")[-1]
	metadata_tuple = filename.split("_")
	metadata.append(metadata_tuple[0])
	image_segment = metadata_tuple[3]
	index_i = image_segment.index("i")
	index_j = image_segment.index("j")
	index_dot = image_segment.index(".jpg")

	metadata.append(int(image_segment[index_i+1:index_j]))
	metadata.append(int(image_segment[index_j+1:index_dot]))
	return (metadata)


def create_label(filepath):
	return get_metadata(filepath)


filenames, classes = _get_filenames_and_classes(dataset_dir=dataset_dir)

print(filenames[0])
filename_queue = tf.train.string_input_producer(filenames)

reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)

key = tf.py_func(create_label, [key], [tf.string])

my_img = tf.image.decode_jpeg(value)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)
	print(my_img.eval())
	print(sess.run(key))
	coord.request_stop()
	coord.join(threads)