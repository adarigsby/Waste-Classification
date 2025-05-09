import json 
import pandas as pd
import csv

json_results_file = "trashnet_waste_classification_results.json"
csv_metrics_file = "trashnet_metrics.csv"

full_trashnet = pd.read_json(json_results_file)
trashnet_without_trash_class = full_trashnet[full_trashnet['actual_class_id'] != 5]

metrics_for_csv = []

def calculate(data):

    metrics = {}
    metrics['dataset'] = data

    image_count = len(data)
    metrics['number of images'] = image_count
    with_prediction = data['predictions'].apply(lambda x: len(x) > 0).sum()
    metrics['number of images with predictions'] = with_prediction
    metrics['number of images without predictions'] = image_count - with_prediction 

    #separating out the images where a prediction was made by the modle, and where a prediction was not made
    data_with_pred = data[data['predictions'].apply(lambda x: len(x) > 0)].copy()
    data_without_pred = data[data['predictions'].apply(lambda x: len(x) == 0)].copy()

    #how many of each trashnet classes
    actual_class_counts = data['actual_class_id'].value_counts().sort_index()
    for actual_class_id, count in actual_class_counts.items():
        metrics[f'actual class {actual_class_id} count'] = count

    #how many of each model class classifications
    classification_counts = data['predictions'].apply(lambda x: x[0]['class_id'] if len(x) > 0 else None).value_counts().sort_index()
    for classification_class_id, count in classification_counts.items():
        metrics[f'class {classification_class_id} classifications'] = count

    metrics['average time'] = data['time'].mean()
    metrics['average time for images with predictions'] = data_with_pred['time'].mean()
    metrics['average time for images without predictions']  = data_without_pred['time'].mean()

    confidence_values = data['predictions'].apply(lambda x: x[0]['confidence'] if len(x) > 0 else None)
    metrics['average confidence'] = confidence_values.mean()

    correct_predictions = (data_with_pred['actual_class_id'] == data_with_pred['predictions'].apply(lambda x: x[0]['class_id'])).sum()
    metrics['number of correct predictions'] = correct_predictions
    metrics['number of incorrect predictions'] = with_prediction - correct_predictions

    #accuracy only looking at images that had predictions 
    metrics['accuracy of images with predictions'] = (correct_predictions / with_prediction) * 100

    #accuracy looking at all images - and counting those without predictions as wrong 
    metrics['accuracy of all images'] = (correct_predictions / image_count) * 100

    #precision per class - only looking at images with predictions 
    class_precisions = []
    class_true_positives = []
    class_false_positives = []

    for i in range(5):
        true_positives = ((data_with_pred['predictions'].apply(lambda x: x[0]['class_id']) == i) & (data_with_pred['actual_class_id'] == i)).sum()
        false_positives = ((data_with_pred['predictions'].apply(lambda x: x[0]['class_id']) == i) & (data_with_pred['actual_class_id'] != i)).sum()

        class_precisions.append(true_positives / (true_positives + false_positives))
        metrics[f'class {i} precision'] = class_precisions[i]
        class_true_positives.append(true_positives)
        class_false_positives.append(false_positives)

    #recall per class - counting no prediction at all as a false negative 
    class_recalls = []
    class_false_negatives = []

    for i in range(5):
        false_negatives_with_pred = ((data_with_pred['actual_class_id'] == i) & (data_with_pred['predictions'].apply(lambda x: x[0]['class_id']) != i)).sum()
        false_negatives_without_pred = (data_without_pred ['actual_class_id'] == i).sum()
        false_negatives = false_negatives_with_pred + false_negatives_without_pred

        class_recalls.append(class_true_positives[i] / (class_true_positives[i] + false_negatives))
        metrics[f'class {i} recall'] = class_recalls[i]
        class_false_negatives.append(false_negatives)

    #F1 per class
    class_F1_values = []

    for i in range(5):
        metrics[f'class {i} F1'] = (2 * class_precisions[i] * class_recalls[i]) / (class_precisions[i] + class_recalls[i])

    metrics_for_csv.append(metrics)

calculate(full_trashnet)
calculate(trashnet_without_trash_class)

with open('trashnet_metrics.csv','w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=metrics_for_csv[0].keys())
    writer.writeheader()
    writer.writerows(metrics_for_csv)
