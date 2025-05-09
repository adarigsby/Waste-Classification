import json 
import pandas as pd
import csv

json_results_file = "resort_it_waste_classification_results.json"

full_resort_it = pd.read_json(json_results_file)
resort_it_without_last_class = full_resort_it[full_resort_it['actual_class_id'] != 6]

metrics_for_csv = []

metrics = {}
metrics['dataset'] = "resort-it"

image_count = len(full_resort_it)
metrics['number of images prior to removing last class'] = image_count
new_image_count = len(resort_it_without_last_class)
metrics['number of images after removing last class'] = new_image_count

with_prediction = resort_it_without_last_class['predictions'].apply(lambda x: len(x) > 0).sum()
metrics['number of images with predictions'] = with_prediction
metrics['number of images without predictions'] = new_image_count - with_prediction 

#separating out the images where a prediction was made by the modle, and where a prediction was not made
data_with_pred = resort_it_without_last_class[resort_it_without_last_class['predictions'].apply(lambda x: len(x) > 0)].copy()
data_without_pred = resort_it_without_last_class[resort_it_without_last_class['predictions'].apply(lambda x: len(x) == 0)].copy()

#how many of each resort it classes (even though the class ids have been converted to match up with the model class ids)
actual_class_counts = resort_it_without_last_class['actual_class_id'].value_counts().sort_index()
for actual_class_id, count in actual_class_counts.items():
    metrics[f'actual class {actual_class_id} count'] = count

#how many of each model class classifications
classification_counts = resort_it_without_last_class['predictions'].apply(lambda x: x[0]['class_id'] if len(x) > 0 else None).value_counts().sort_index()
for classification_class_id, count in classification_counts.items():
    metrics[f'class {classification_class_id} classifications'] = count

metrics['average time'] = resort_it_without_last_class['time'].mean()
metrics['average time for images with predictions'] = data_with_pred['time'].mean()
metrics['average time for images without predictions']  = data_without_pred['time'].mean()

confidence_values = resort_it_without_last_class['predictions'].apply(lambda x: x[0]['confidence'] if len(x) > 0 else None)
metrics['average confidence'] = confidence_values.mean()

initial_correct_predictions = (data_with_pred['actual_class_id'] == data_with_pred['predictions'].apply(lambda x: x[0]['class_id'])).sum()
additional_correct_predictions = ((data_with_pred['actual_class_id'] == 3) & (data_with_pred['predictions'].apply(lambda x: x[0]['class_id']) == 0)).sum()
correct_predictions = initial_correct_predictions + additional_correct_predictions
metrics['number of correct predictions'] = correct_predictions
metrics['number of incorrect predictions'] = with_prediction - correct_predictions

#accuracy only looking at images that had predictions 
metrics['accuracy of images with predictions'] = (correct_predictions / with_prediction) * 100

#accuracy looking at all images - and counting those without predictions as wrong 
metrics['accuracy of all images'] = (correct_predictions / new_image_count) * 100

#precision per class - only looking at images with predictions 
class_precisions = []
class_true_positives = []
class_false_positives = []

#if the classification is 0 (cardboard) and the actual class id is paper (3) - this is a true positive because many of the paper photos are actually cardboard
for i in range(5):
    if i == 3: 
        true_positives = (((data_with_pred['predictions'].apply(lambda x: x[0]['class_id']) == 0) | (data_with_pred['predictions'].apply(lambda x: x[0]['class_id']) == 3))  & (data_with_pred['actual_class_id'] == 3)).sum()
        false_positives = (((data_with_pred['predictions'].apply(lambda x: x[0]['class_id']) == 0) | (data_with_pred['predictions'].apply(lambda x: x[0]['class_id']) == 3)) & (data_with_pred['actual_class_id'] != 3)).sum()
    else:
        true_positives = ((data_with_pred['predictions'].apply(lambda x: x[0]['class_id']) == i) & (data_with_pred['actual_class_id'] == i)).sum()
        false_positives = ((data_with_pred['predictions'].apply(lambda x: x[0]['class_id']) == i) & (data_with_pred['actual_class_id'] != i)).sum()

    class_precisions.append(true_positives / (true_positives + false_positives))
    metrics[f'class {i} precision'] = class_precisions[i]
    class_true_positives.append(true_positives)
    class_false_positives.append(false_positives)

#recall per class - counting no prediction at all as a false negative 
#when the actual class if is 3, it is only a false negative does not equal 3 and does not equal 0 
class_recalls = []
class_false_negatives = []

for i in range(5):
    if i == 3:
        false_negatives_with_pred = ((data_with_pred['actual_class_id'] == 3) & (data_with_pred['predictions'].apply(lambda x: x[0]['class_id'])!= 3) & (data_with_pred['predictions'].apply(lambda x: x[0]['class_id'])!= 0)).sum()
    else: 
        false_negatives_with_pred = ((data_with_pred['actual_class_id'] == i) & (data_with_pred['predictions'].apply(lambda x: x[0]['class_id']) != i)).sum()

    false_negatives_without_pred = (data_without_pred['actual_class_id'] == i).sum()
    false_negatives = false_negatives_with_pred + false_negatives_without_pred

    if (class_true_positives[i] + false_negatives) > 0:
        class_recalls.append(class_true_positives[i] / (class_true_positives[i] + false_negatives))
    else: 
        class_recalls.append(0)
        
    metrics[f'class {i} recall'] = class_recalls[i]
    class_false_negatives.append(false_negatives)

class_F1_values = []
for i in range(5):
    if (class_precisions[i] + class_recalls[i]) > 0:
        metrics[f'class {i} F1'] = (2 * class_precisions[i] * class_recalls[i]) / (class_precisions[i] + class_recalls[i])
    else: 
        metrics[f'class {i} F1'] = 0

metrics_for_csv.append(metrics)

with open('resort_it_metrics.csv','w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=metrics_for_csv[0].keys())
    writer.writeheader()
    writer.writerows(metrics_for_csv)
