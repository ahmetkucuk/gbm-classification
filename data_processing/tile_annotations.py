import os
from scipy import misc
from PIL import Image, ImageDraw


def read_image(image_path):
	return misc.imread(image_path, flatten=False)


def read_annotation(filename):
	annotations = []
	with open(filename, 'r') as f:
		for line in f.readlines():
			if "x" in line:
				continue
			values = line.replace('\n', '').split('\t')
			values = [int(v) for v in values]
			values[2] += values[0]
			values[3] += values[1]
			annotations.append(values)
	return annotations


def files_in_dir(directory):
	result = []
	for file in os.listdir(directory):
		if file.endswith("-0.png"):
			result.append(file)
	return result


def tile_annotation(image, output_dir, image_name, annotation, annotation_index):

	image_name = image_name[0:12] + "annotation" + str(annotation_index) + "_"
	x1 = annotation[0]
	y1 = annotation[1]
	x2 = annotation[2]
	y2 = annotation[3]
	image = image[x1:x2, y1:y2]

	w = image.shape[0]
	h = image.shape[1]
	for i in range(int(w/256)):
		for j in range(int(h/256)):
			output_path = os.path.join(output_dir, image_name + "i" + format(i, '03d') + "j" + format(j, '03d') + ".jpg")
			patch = image[i*256:i*256+256, j*256:j*256+256]
			patch = Image.fromarray(patch)
			patch.save(output_path)


image_dir = "/home/ahmet/workspace/tcga/extracted_png/"
output_dir = "/home/ahmet/workspace/tcga/annotated_patches/"
annotation_dir = "/home/ahmet/workspace/tcga/annotations/"

image_files = files_in_dir(image_dir)

for i_file in image_files:
	image_path = os.path.join(image_dir, i_file)
	output_path = os.path.join(output_dir, i_file)
	annotation_path = os.path.join(annotation_dir, i_file.replace('-0.png', '.txt'))
	annotations = read_annotation(annotation_path)

	img_arr = read_image(image_path)

	annotation_index = 0
	for a in annotations:
		tile_annotation(img_arr, output_dir, i_file, a, annotation_index)
		annotation_index += 1

