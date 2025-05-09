import json 
import pandas as pd
import csv

json_results_file = "mju_waste_classification_results.json"

full_mju = pd.read_json(json_results_file)

metrics_for_csv = []

metrics = {}
metrics['dataset'] = "MJU"

image_count = len(full_mju)
metrics['number of images'] = image_count
with_prediction = full_mju['predictions'].apply(lambda x: len(x) > 0).sum()
metrics['number of images with predictions'] = with_prediction
metrics['number of images without predictions'] = image_count - with_prediction 

#separating out the images where a prediction was made by the modle, and where a prediction was not made
data_with_pred = full_mju[full_mju['predictions'].apply(lambda x: len(x) > 0)].copy()
data_without_pred = full_mju[full_mju['predictions'].apply(lambda x: len(x) == 0)].copy()

#how many of each model class classifications
classification_counts = full_mju['predictions'].apply(lambda x: x[0]['class_id'] if len(x) > 0 else None).value_counts().sort_index()
for classification_class_id, count in classification_counts.items():
    metrics[f'class {classification_class_id} classifications'] = count

metrics['average time'] = full_mju['time'].mean()
metrics['average time for images with predictions'] = data_with_pred['time'].mean()
metrics['average time for images without predictions']  = data_without_pred['time'].mean()

confidence_values = full_mju['predictions'].apply(lambda x: x[0]['confidence'] if len(x) > 0 else None)
metrics['average confidence'] = confidence_values.mean()

metrics_for_csv.append(metrics)

with open('mju_metrics.csv','w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=metrics_for_csv[0].keys())
    writer.writeheader()
    writer.writerows(metrics_for_csv)
