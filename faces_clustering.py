import optics
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.cluster import DBSCAN

encodings_path = "face-clustering/encodings.pickle"

# load the serialized face encodings + bounding box locations from
# disk, then extract the set of encodings to so we can cluster on
# them
print(">> Loading encodings...")
data = pickle.loads(open(encodings_path, "rb").read())
data = np.array(data)
encodings = [d["encoding"] for d in data]

# cluster the embeddings
print(">> Clustering...")
clusters, noise = optics.template_clustering(encodings, 10, 10)
print(clusters, noise)

labelIDs = np.unique(clusters)
print(labelIDs)

# cluster the embeddings
print("[INFO] clustering...")
clt = DBSCAN(metric="euclidean", n_jobs=4)
clt.fit(encodings)

print(clt.labels_)

# determine the total number of unique faces found in the dataset
labelIDs = np.unique(clt.labels_)
print(labelIDs)
numUniqueFaces = len(np.where(labelIDs > -1)[0])

print("[INFO] # unique faces: {}".format(numUniqueFaces))
