
import numpy as np
from scipy import misc
import os
from shutil import copy2


def read_image(image_path):
	return misc.imread(image_path, flatten=True)


def files_in_dir(directory):
	result = []
	for file in os.listdir(directory):
		if file.endswith(".jpg"):
			result.append(file)
	return result


image_dir = "/home/ahmet/workspace/tcga/patches1k/"
output_dir = "/home/ahmet/workspace/tcga/patches1k_extracted/"

image_names = files_in_dir(image_dir)
for n in image_names:
	path_to_image = os.path.join(image_dir, n)
	image_array = read_image(path_to_image)
	print(np.mean(image_array))
	if np.mean(image_array) < 200:
		copy2(path_to_image, output_dir)