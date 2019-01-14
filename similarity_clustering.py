import optics
import pickle
import cv2
import numpy as np

from imutils import build_montages

if __name__ == "__main__":
    encodings_path = "Ressources/GUFD_encodings.pickle"

    print(">> Loading encodings...")
    data = pickle.loads(open(encodings_path, "rb").read())
    data = np.array(data)
    encodings = [d["encoding"] for d in data]

    max_val = max(encodings)[0]

    '''
    for encoding in encodings:
        encoding[0] = encoding[0] / max_val
    '''

    best = 0
    best_idx = 0
    for i in range(2, 15):
        clust = optics.cluster(encodings, min_samples=i ,reachability_plot=False, clustering_visualization=False)
        labelIDs = np.unique(clust.labels_)
        numClusters = len(np.where(labelIDs > -1)[0])
        if numClusters > best:
            best = numClusters
            best_idx = i

    print(">> Using min samples: " + str(best))

    clust = optics.cluster(encodings, min_samples=best ,reachability_plot=True, clustering_visualization=False)

    # determine the total number of unique faces found in the dataset
    labelIDs = np.unique(clust.labels_)
    numClusters = len(np.where(labelIDs > -1)[0])
    print(">> # of clusters: {}".format(numClusters))

	# loop over the unique face integers
    for labelID in labelIDs:
        # find all indexes into the `data` array that belong to the
        # current label ID, then randomly sample a maximum of 25 indexes
        # from the set
        print(">> Faces for cluster ID: {}".format(labelID))
        idxs = np.where(clust.labels_ == labelID)[0]
        idxs = np.random.choice(idxs, size=min(25, len(idxs)),
        replace=False)

        # initialize the list of faces to include in the montage
        faces = []

        # loop over the sampled indexes
        for i in idxs:
            # load the input image and extract the face ROI
            image = cv2.imread(data[i]["imagePath"])

            # force resize the face ROI to 96x96 and then add it to the
            # faces montage list
            face = cv2.resize(image, (96, 96))
            faces.append(face)

        # create a montage using 96x96 "tiles" with 5 rows and 5 columns
        montage = build_montages(faces, (96, 96), (5, 5))[0]

        # show the output montage
        title = "Cluster #{}".format(labelID)
        title = "Not clustered" if labelID == -1 else title
        cv2.imshow(title, montage)
        cv2.waitKey(0)