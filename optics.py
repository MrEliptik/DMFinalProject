from pyclustering.cluster.optics import optics, ordering_analyser, ordering_visualizer
from pyclustering.cluster import cluster_visualizer_multidim
from pyclustering.utils import read_sample, timedcall
from pyclustering.samples.definitions import SIMPLE_SAMPLES, FCPS_SAMPLES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def template_clustering(sample, eps, minpts, amount_clusters=None, visualize=True, ccore=False):
    optics_instance = optics(sample, eps, minpts, amount_clusters, ccore)
    optics_instance.process()

    if (visualize is True):
        clusters = optics_instance.get_clusters()
        noise = optics_instance.get_noise()

        #visualizer = cluster_visualizer_multidim()
        #visualizer.append_clusters(clusters, sample)
        #visualizer.append_cluster(noise, sample, marker='x')
        #visualizer.show()

        ordering = optics_instance.get_ordering()
        analyser = ordering_analyser(ordering)

        print(analyser.extract_cluster_amount(eps))

        ordering_visualizer.show_ordering_diagram(analyser, amount_clusters)

        objects = optics_instance.get_optics_objects()

        return clusters, noise
if __name__ == "__main__":
    df = pd.read_csv("Datasets/dataset_diabetes/diabetic_data.csv", usecols=[6, 9, 16])

    admissions = np.asarray(df["admission_type_id"])
    emergencies = np.asarray(df["number_emergency"])
    time_hospital = np.asarray(df["time_in_hospital"])

    plt.scatter(admissions, emergencies)
    #plt.show()

    plt.scatter(time_hospital, emergencies)
    #plt.show()

    template_clustering(list(time_hospital), 5, 4)