from scipy import misc
from PIL import Image, ImageDraw
import numpy as np
import os


def read_image(image_path):
	return misc.imread(image_path, flatten=True)


def draw_rectangle(draw, coordinates, color, width=1):
	for i in range(width):
		rect_start = (coordinates[0] - i, coordinates[1] - i)
		rect_end = (coordinates[2] + i, coordinates[3] + i)
		draw.rectangle((rect_start, rect_end), outline=color)


def files_in_dir(directory):
	result = []
	for file in os.listdir(directory):
		if file.endswith("-6.png"):
			result.append(file)
	return result


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


image_dir = "/home/ahmet/workspace/tcga/extracted_png/"
output_dir = "/home/ahmet/workspace/tcga/annotated_png/"
annotation_dir = "/home/ahmet/workspace/tcga/annotations/"

image_files = files_in_dir(image_dir)

for i_file in image_files:
	image_path = os.path.join(image_dir, i_file)
	output_path = os.path.join(output_dir, i_file)
	annotation_path = os.path.join(annotation_dir, i_file.replace('-6.png', '.txt'))
	annotations = read_annotation(annotation_path)

	pil_image = Image.open(image_path)
	pil_image.load()
	draw = ImageDraw.Draw(pil_image)

	for a in annotations:
		draw_rectangle(draw, a, 'red', 10)

	pil_image.save(output_path)