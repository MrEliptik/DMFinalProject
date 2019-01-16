import pandas as pd
import matplotlib.pyplot as plt
import optics
import numpy as np
import csv

if __name__ == "__main__":  
    df = pd.read_csv("Datasets/dataset_diabetes/diabetic_data.csv", usecols = [3,4,9,12,16])

    values = df.values.tolist()

    for val in values:
        # Convert gender
        if val[0] == "Male":
            val[0] = 0
        else:
            val[0] = 1

        # Convert age to class 0 to 9
        for i in range(0, 100):
            if ("[" + str(i) + "-" + str(i+10) + ")") == val[1]:
                val[1] = i

    #clust = optics.cluster(values[100000:], min_samples=200, eps=10.2)
    clust = optics.cluster(values[100000:], min_samples=250, eps=6)

    # determine the total number of unique faces found in the dataset
    labelIDs = np.unique(clust.labels_)
    numClusters = len(np.where(labelIDs > -1)[0])
    print(">> # clusters: {}".format(numClusters))

    cluster_insights = []

    for labelID in labelIDs:
        # find all indexes into the `data` array that belong to the
        # current label ID, then randomly sample a maximum of 25 indexes
        # from the set
        print(">> Values for cluster ID: {}".format(labelID))
        idxs = np.where(clust.labels_ == labelID)[0]

        male            = 0
        female          = 0
        age             = 0
        time_hosp       = 0
        nb_proc         = 0
        nb_emergencies  = 0
        nb_sample       = len(idxs)
        # loop over the sampled indexes
        for i in idxs:
            # load the input image and extract the face ROI
            #print(values[i])
            if values[i][0]:
                female      += 1
            else:
                male        += 1
            
            age             += values[i][1]
            time_hosp       += values[i][2]
            nb_proc         += values[i][3]
            nb_emergencies  += values[i][4]

        insights = {
                "ClusterID": labelID,
                "Male%": male * 100 / nb_sample,
                "Female%": female * 100 / nb_sample,
                "AgeClass": age / nb_sample,
                "TimeInHospital": time_hosp / nb_sample,
                "NbOfProcedures": nb_proc / nb_sample,
                "NbOfEmergencies": nb_emergencies / nb_sample,
                }
        cluster_insights.append(insights)

        print(insights)


    with open('Results/Diabetes/clusters.csv', mode='w') as csv_file:
            fieldnames = ['ClusterID','Male%', 'Female%', 'AgeClass', 'TimeInHospital', 'NbOfProcedures', 'NbOfEmergencies']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()  
            for cluster in cluster_insights:
                writer.writerow(cluster)

        


