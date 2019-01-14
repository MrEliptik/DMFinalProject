import cv2
import os
import pickle
import numpy as np
import imutils
import dlib

from scipy.spatial import distance
from imutils import paths
from imutils import face_utils


def getFacialFeatures(img, visualize=False):
	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor("Ressources/shape_predictor_68_face_landmarks.dat")

	# load the input image, resize it, and convert it to grayscale
	image = img
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale image
	rects = detector(gray, 1)

	# loop over the face detections
	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the landmark (x, y)-coordinates to a NumPy array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		if(visualize):
			# loop over the face parts individually
			for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
				# clone the original image so we can draw on it, then
				# display the name of the face part on the image
				clone = image.copy()
				cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
					0.7, (0, 0, 255), 2)

				# loop over the subset of facial landmarks, drawing the
				# specific face part
				for (x, y) in shape[i:j]:
					cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

				# extract the ROI of the face region as a separate image
				(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
				roi = image[y:y + h, x:x + w]
				roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)

				# show the particular face part
				cv2.imshow("ROI", roi)
				cv2.imshow("Image", clone)
				cv2.waitKey(0)

			# visualize all facial landmarks with a transparent overlay
			output = face_utils.visualize_facial_landmarks(image, shape)
			cv2.imshow("Image", output)
			cv2.waitKey(0)

	return face_utils.FACIAL_LANDMARKS_IDXS, shape

def extractJawFeatures(shape):
    # Get the jaw features
    jaw_left 		= shape[17 - 1]
    jaw_right 		= shape[1 - 1]
    jaw_bottom 		= shape[9 - 1]
    jaw_width 		= distance.euclidean(jaw_left, jaw_right)

    return jaw_width

def extractNoseFeatures(shape, normalizer):
    # Get the nose features
    nose_top 		= shape[28 - 1]
    nose_left 		= shape[36 - 1]
    nose_right 		= shape[32 - 1]
    nose_bottom 	= shape[34 - 1]
    nose_width 		= distance.euclidean(nose_left, nose_right)
    nose_height 	= distance.euclidean(nose_top, nose_bottom)
    nose_ratio 		= nose_height / nose_width
    nose_size 		= nose_height / normalizer

    return nose_ratio, nose_size

def extractEyeFeatures(shape, normalizer):
    # Get the eye features
    (i, j) 				= features["left_eye"]
    left_eye_top 		= shape[45 - 1]
    left_eye_left 		= shape[46 - 1]
    left_eye_right 		= shape[43 - 1]
    left_eye_bottom 	= shape[47 - 1]
    left_eye_width 		= distance.euclidean(left_eye_left, left_eye_right)
    left_eye_height 	= distance.euclidean(left_eye_top, left_eye_bottom)
    left_eye_ratio 		= left_eye_height / left_eye_width
    left_eye_size 		= left_eye_height / normalizer
    left_eye_distance	= left_eye_width / normalizer

    (i, j) 				= features["right_eye"]
    right_eye_top 		= shape[38 - 1]
    right_eye_left 		= shape[40 - 1]
    right_eye_right 	= shape[37 - 1]
    right_eye_bottom 	= shape[42 - 1]
    right_eye_width 	= distance.euclidean(right_eye_left, right_eye_right)
    right_eye_height 	= distance.euclidean(right_eye_top, right_eye_bottom)
    right_eye_ratio 	= right_eye_height / right_eye_width
    right_eye_size 		= right_eye_height / normalizer
    right_eye_distance	= right_eye_width / normalizer

    eye_size			= ((right_eye_size + left_eye_size) / 2) / normalizer
    eye_distance		= ((right_eye_distance + left_eye_distance) / 2) / normalizer

    return eye_size, eye_distance, [right_eye_top, left_eye_top]

def extractEyebrowFeatures(shape, eyeFeatures, normalizer):
    right_eye_top   = eyeFeatures[0]
    left_eye_top    = eyeFeatures[1]

    # Get the eyebrow features
    left_eyebrow_top 		= shape[25 - 1]
    left_eyebrow_left 		= shape[27 - 1]
    left_eyebrow_right 		= shape[23 - 1]
    left_eyebrow_width 		= distance.euclidean(left_eyebrow_left, left_eyebrow_right)
    left_eyebrow_distance	= left_eyebrow_width / normalizer

    # Get the eyebrow features
    (i, j) 					= features["right_eyebrow"]
    right_eyebrow_top 		= shape[20 - 1]
    right_eyebrow_left 		= shape[22 - 1]
    right_eyebrow_right 	= shape[18 - 1]
    right_eyebrow_width 	= distance.euclidean(right_eyebrow_left, right_eyebrow_right)
    right_eyebrow_distance	= right_eyebrow_width / normalizer

    eyebrow_width 			= ((right_eyebrow_width + left_eyebrow_width) / 2) / normalizer
    eyebrow_lift			= ((distance.euclidean(right_eyebrow_top, right_eye_top) + 
                                distance.euclidean(left_eyebrow_top, left_eye_top)) / 2) / normalizer

    return eyebrow_width, eyebrow_lift

if __name__ == "__main__":  
    dataset_path    = "Datasets/GUFD/"
    encodings_path  = "Ressources/GUFD_encodings.pickle"

    imagePaths = list(paths.list_images(dataset_path))

    features_dataset = []

    for (i, imagePath) in enumerate(imagePaths):
        print(">> Processing image {}/{}".format(i + 1, len(imagePaths)))
        print(" " + imagePath)
        im = cv2.imread(imagePath)

        # Extract the facial feature
        print(" " + "Extracting features..")
        features, shape                     = getFacialFeatures(im)
        jaw_width                           = extractJawFeatures(shape)
        nose_ratio, nose_size               = extractNoseFeatures(shape, jaw_width)
        eye_size, eye_distance, eyeFeatures = extractEyeFeatures(shape, jaw_width)
        eyebrow_width, eyebrow_lift         = extractEyebrowFeatures(shape, eyeFeatures, jaw_width)

        # build a dictionary of the image path, bounding box location,
        # and facial encodings for the current image
        d = [{
                "imagePath": imagePath, 
                "encoding": [jaw_width, nose_ratio, nose_size, eye_size, eye_distance, eyebrow_width, eyebrow_lift]
            }]
        features_dataset.extend(d)

    # dump the facial encodings data to disk
    print(">> Serializing encodings...")
    f = open(encodings_path, "wb")
    f.write(pickle.dumps(features_dataset))
    f.close()
