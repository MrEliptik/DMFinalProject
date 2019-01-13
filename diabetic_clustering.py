import pandas as pd
import matplotlib.pyplot as plt
import optics

df = pd.read_csv("Datasets/dataset_diabetes/diabetic_data.csv", usecols = [4,12])


matrix = []

for i in range(0, 100, 10):
    age_range = "[" + str(i) + "-" + str(i+10) + ")"
    values = df[df.age == age_range].num_lab_procedures.values
    for value in values:
        matrix.append([value,i])

optics.cluster(matrix[90000:])

