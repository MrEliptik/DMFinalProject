import optics
import pickle
import numpy as np

if __name__ == "__main__":
    encodings_path = "Ressources/GUFD_encodings.pickle"

    print(">> Loading encodings...")
    data = pickle.loads(open(encodings_path, "rb").read())
    data = np.array(data)
    encodings = [d["encoding"] for d in data]

    best = 0
    best_idx = 0
    for i in range(2, 15):
        clust = optics.cluster(encodings, min_samples=i ,reachability_plot=False, clustering_visualization=False)
        labelIDs = np.unique(clust.labels_)
        numUniqueFaces = len(np.where(labelIDs > -1)[0])
        if numUniqueFaces > best:
            best = numUniqueFaces
            best_idx = i

    clust = optics.cluster(encodings, min_samples=best ,reachability_plot=True, clustering_visualization=False)