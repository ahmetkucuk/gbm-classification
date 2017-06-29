import os
from scipy import misc
from PIL import Image, ImageDraw


def read_image(image_path):
	return misc.imread(image_path, flatten=False)


def files_in_dir(directory):
	result = []
	for file in os.listdir(directory):
		if file.endswith("-0.png"):
			result.append(file)
	return result


def tile_image(image, output_dir, image_name):
	w = image.shape[0]
	h = image.shape[1]
	image_name = image_name[0:12] + "_"
	for i in range(int(w/1024)):
		for j in range(int(h/1024)):
			output_path = os.path.join(output_dir, image_name + "i" + format(i, '03d') + "j" + format(j, '03d') + ".jpg")
			patch = image[i*1024:i*1024+1024, j*1024:j*1024+1024]
			patch = Image.fromarray(patch)
			patch.save(output_path)


image_dir = "/home/ahmet/workspace/tcga/extracted_png/"
output_dir = "/home/ahmet/workspace/tcga/patches1k/"
image_files = files_in_dir(image_dir)


for i_file in image_files:
	image = read_image(os.path.join(image_dir, i_file))
	tile_image(image, output_dir, i_file)

