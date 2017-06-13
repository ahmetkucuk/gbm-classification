import numpy as np
from PIL import Image
import os


class PatchPosition(object):

	def __init__(self, row, col):
		self.row = row
		self.col = col

	def __hash__(self):
		return hash((self.row, self.col))

	def __eq__(self, other):
		return (self.row, self.col) == (other.row, other.col)

	def __ne__(self, other):
		return not(self == other)


class SlidesPredictionHolder(object):

	def __init__(self):
		self.slide_prediction_holders_by_id = {}

	def set_predictions(self, probs, labels, file_ids, rows, cols):

		for p, l, id, r, c in zip(probs, labels, file_ids, rows, cols):

			if id not in self.slide_prediction_holders_by_id.keys():
				self.slide_prediction_holders_by_id[id] = {}

			key = self.create_key(r, c)
			slide_x = self.slide_prediction_holders_by_id[id]
			slide_x[key] = p

	def create_key(self, row, col):
		return PatchPosition(row, col)

	def get_predictions(self):
		return self.slide_prediction_holders_by_id

	def build_heat_map(self, output_dir, epochs):

		for slide in self.slide_prediction_holders_by_id.keys():
			predictions_by_patch_position = self.slide_prediction_holders_by_id[slide]
			max_x = 0
			max_y = 0
			for position in predictions_by_patch_position.keys():
				x = position.row
				y = position.col
				if x > max_x:
					max_x = x
				if y > max_y:
					max_y = y

			image = [[0 for i in range(max_y+1)] for j in range(max_x+1)]
			heatmap_image = [[0 for i in range(max_y+1)] for j in range(max_x+1)]
			for position in predictions_by_patch_position.keys():
				predictions = predictions_by_patch_position[position]

				heatmap_image[x][y] = np.max(predictions) * 1.0
				if np.max(predictions) > 0.9:
					image[x][y] = np.max(predictions)
				else:
					image[x][y] = 0

			img = Image.fromarray(np.array(image), 'RGB')
			img.save(os.path.join(output_dir, slide + '_predictions_epochs_' + str(epochs) + '.png'))

			heatmap_image = np.array(heatmap_image)
			heatmap_image = ((heatmap_image - np.min(heatmap_image)) / (np.max(heatmap_image) - np.min(heatmap_image))*255)
			heatmap_image = Image.fromarray(heatmap_image, 'RGB')
			heatmap_image.save(os.path.join(output_dir, slide + '_heatmap_epochs_' + str(epochs) + '.png'))