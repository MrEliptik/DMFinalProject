from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os


if __name__ == "__main__":	
	dataset_path 	= "Datasets/footballers/"
	encodings_path 	= "Ressources/footballers_encodings.pickle"

	# For predictions
	#dataset_path 	= "Datasets/footballers/Predict/"
	#encodings_path 	= "Ressources/footballers_predict_encodings.pickle"

	# grab the paths to the input images in our dataset, then initialize
	# out data list (which we'll soon populate)
	print(">> Quantifying faces...")
	imagePaths = list(paths.list_images(dataset_path))
	data = []

	# loop over the image paths
	for (i, imagePath) in enumerate(imagePaths):
		# load the input image and convert it from RGB (OpenCV ordering)
		# to dlib ordering (RGB)
		print(">> Processing image {}/{}".format(i + 1, len(imagePaths)))
		print(imagePath)
		image = cv2.imread(imagePath)
		rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		# detect the (x, y)-coordinates of the bounding boxes
		# corresponding to each face in the input image
		boxes = face_recognition.face_locations(rgb, model="cnn")

		# compute the facial embedding for the face
		encodings = face_recognition.face_encodings(rgb, boxes)

		# build a dictionary of the image path, bounding box location,
		# and facial encodings for the current image
		d = [{"imagePath": imagePath, "loc": box, "encoding": enc}
			for (box, enc) in zip(boxes, encodings)]
		data.extend(d)

	# dump the facial encodings data to disk
	print(">> Serializing encodings...")
	f = open(encodings_path, "wb")
	f.write(pickle.dumps(data))
	f.close()