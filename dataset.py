import os
import tensorflow as tf
from scipy import misc
import numpy as np
from PIL import Image

from sklearn.utils import shuffle


def get_image(imagename):
	temp = Image.open(imagename)
	keep = temp.copy()
	temp.close()
	return keep


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


def get_metadata(filenames):

	metadata_list = []
	for filename in filenames:
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
		metadata_list.append(metadata)
	return metadata_list


def get_datasets(dataset_dir, image_size):

	filenames, class_names = _get_filenames_and_classes(dataset_dir=dataset_dir)

	data = []
	labels = []
	for i in filenames:
		data.append(get_image(i))
		if "/AII/" in i:
			labels.append([1, 0, 0, 0, 0, 0, 0, 0])
		elif "/AIII/" in i:
			labels.append([0, 1, 0, 0, 0, 0, 0, 0])
		elif "/OII/" in i:
			labels.append([0, 0, 1, 0, 0, 0, 0, 0])
		elif "/OIII/" in i:
			labels.append([0, 0, 0, 1, 0, 0, 0, 0])
		elif "/OAII/" in i:
			labels.append([0, 0, 0, 0, 1, 0, 0, 0])
		elif "/OAIII/" in i:
			labels.append([0, 0, 0, 0, 0, 1, 0, 0])
		elif "/GBM/" in i:
			labels.append([0, 0, 0, 0, 0, 0, 1, 0])
		else:
			labels.append([0, 0, 0, 0, 0, 0, 0, 1])

	print("Finished Reading images")
	data = preprocess(data, image_size)
	metadata = get_metadata(filenames)
	data, labels, metadata = shuffle(data, labels, metadata)
	split_at = int(len(data) * 0.8)
	train_files = data[:split_at]
	train_labels = labels[:split_at]
	train_metadata = metadata[:split_at]

	val_files = data[split_at:]
	val_labels = labels[split_at:]
	val_metadata = metadata[split_at:]

	return TissueDataset(data=train_files, labels=train_labels, metadata=train_metadata), TissueDataset(data=val_files, labels=val_labels, metadata=val_metadata)


def preprocess(images, image_size):

	processed_images = []
	for i in images:
		im = i.convert("L")
		im = im.resize(size=(image_size, image_size))
		im_array = np.array(im)
		im_array = np.expand_dims(im_array, axis=2)
		processed_images.append(im_array)
	return processed_images


class TissueDataset(object):

	def __init__(self, data, labels, metadata):
		self.data = data
		self.labels = labels
		self.batch_index = 0
		self.metadata = metadata
		self.preds = [[0.1, 0.2] for i in range(len(data))]

	def next_batch(self, batch_size):
		if self.batch_index*batch_size + batch_size > len(self.data):
			self.batch_index = 0
		batched_data, batched_labels = self.data[self.batch_index*batch_size: self.batch_index*batch_size + batch_size], self.labels[self.batch_index*batch_size: self.batch_index*batch_size + batch_size]
		self.batch_index += 1
		return batched_data, batched_labels, self.batch_index-1

	def set_preds(self, index, batch_size, predictions):

		start_index = index*batch_size
		end_index = index*batch_size + batch_size
		if len(self.preds) < end_index:
			return
		for (i, k) in zip(range(start_index, end_index), range(batch_size)):
			self.preds[i] = predictions[k]

	def get_metedata_and_preds(self):
		return self.metadata, self.preds

	def size(self):
		return len(self.data)

	def build_heat_map(self, output_dir):

		metadata_map = {}
		for (m, p) in zip(self.metadata, self.preds):
			slide_id = m[0]
			if slide_id not in metadata_map:
				metadata_map[slide_id] = []

			metadata_map[slide_id].append([m[1:3], p])

		for slide in metadata_map.keys():
			patch_location_and_predictions = metadata_map[slide]
			max_x = 0
			max_y = 0
			for location in patch_location_and_predictions:
				print(location)
				x = location[0][0]
				y = location[0][1]
				if x > max_x:
					max_x = x
				if y > max_y:
					max_y = y

			print(max_x)
			print(max_y)

			image = [[0 for i in range(max_y+1)] for j in range(max_x+1)]
			for location in patch_location_and_predictions:
				x = location[0][0]
				y = location[0][1]

				if np.max(location) > 0.9:
					image[x][y] = 255
				else:
					image[x][y] = 0

			img = Image.fromarray(np.array(image), 'RGB')
			img.save(output_dir + slide + '_predictions.png')

# train, test = get_datasets(dataset_dir="/Users/ahmetkucuk/Documents/Research/Medical/patches/", image_size=256)
