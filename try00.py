import numpy as np
import csv
import pandas as pd
'''
ground_truth_file = "/home/runge/project/pulse/data/groundTruth.csv"
ground_truth_data = list(np.loadtxt(ground_truth_file, str, delimiter='\n'))
names = ground_truth_data[0].split(";")
ids = ground_truth_data[1].split(";")
result_storage_location = "/home/runge/project/pulse/data/ground_truths.csv"

with open(result_storage_location, 'w') as result_file:
    writer = csv.writer(result_file)
    writer.writerow(["person_id_activity", "ground_truth"])
    for id, value in zip(names, ids):
        writer.writerow([id, value])
        
'''

