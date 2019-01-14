import detect_face_parts
import cv2
from scipy.spatial import distance
import os
from imutils import paths

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


dataset_path = "Datasets/GUFD/"

imagePaths = list(paths.list_images(dataset_path))

features_dataset = []

for (i, imagePath) in enumerate(imagePaths):
    print(">> Processing image {}/{}".format(i + 1, len(imagePaths)))
    print(" " + imagePath)
    im = cv2.imread(imagePath)

    # Extract the facial feature
    print(" " + "Extracting features..")
    features, shape                     = detect_face_parts.getFacialFeatures(im)
    jaw_width                           = extractJawFeatures(shape)
    nose_ratio, nose_size               = extractNoseFeatures(shape, jaw_width)
    eye_size, eye_distance, eyeFeatures = extractEyeFeatures(shape, jaw_width)
    eyebrow_width, eyebrow_lift         = extractEyebrowFeatures(shape, eyeFeatures, jaw_width)

    features_dataset.append([jaw_width, nose_ratio, nose_size, eye_size, eye_distance, eyebrow_width, eyebrow_lift])

'''
jaw_width       /,
eye_distance,   /
nose_ratio,     /
nose_size,      /
eye_size,       /
eyebrow_width,  /
eyebrow_lift    /
'''