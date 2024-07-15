import sys
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
import argparse

os.chdir('/home/iustin/Mech-Interp/Automatic-Circuit-Discovery/acdc') 

def main(input_file, output_dir):
    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Load metrics from the JSON file
    with open(input_file, 'r') as f:
        metrics_data = json.load(f)

    # Extract true labels and predicted scores
    current_metrics = np.array(metrics_data['current_metrics'])
    predicted_scores = np.array(metrics_data['results'])

    # Print statistics for debugging
    print(f"Processing {input_file}")
    print(f"Current Metrics: {current_metrics[:5]}")
    print(f"Predicted Scores: {predicted_scores[:5]}")
    print(f"Current Metrics Range: {current_metrics.min()} - {current_metrics.max()}")
    print(f"Predicted Scores Range: {predicted_scores.min()} - {predicted_scores.max()}")

    # Determine a threshold to create binary labels (using median here for example)
    threshold = np.median(current_metrics)
    true_labels = (current_metrics >= threshold).astype(int)

    # Check unique values in true_labels
    unique_labels = np.unique(true_labels)
    print(f"Unique True Labels: {unique_labels}")

    # Print statistics for verification
    print(f"True Labels Distribution: {np.bincount(true_labels)}")
    print(f"Predicted Scores Mean: {predicted_scores.mean()}")

    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(true_labels, predicted_scores)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # Save the plot as a PNG file
    filename_suffix = os.path.basename(os.path.dirname(os.path.dirname(input_file)))
    output_file = os.path.join(output_dir, f'roc_curve_{filename_suffix}.png')
    plt.savefig(output_file)
    print(f"ROC curve saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate ROC curve from JSON metrics file')
    parser.add_argument('--input-file', type=str, required=True, help='Path to the input JSON file with metrics')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save the ROC curve plot')
    
    args = parser.parse_args()
    main(args.input_file, args.output_dir)