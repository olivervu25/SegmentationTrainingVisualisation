import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd

"""
Confusion Matrix Visualization for Model Performance Evaluation

This script provides functions to visualize and update the confusion matrix of an image 
segmentation model's performance using Matplotlib.

Primary Functions:
- `create_initial_confusion_matrix()`: 
    - Initializes a 2x2 confusion matrix plot.
    - The matrix values are initialized to zero.
    - Configures visual elements such as color mapping, labels, and title.
    - Returns the figure, axis, color axis, and text elements for updating.

- `update_confusion_matrix(ax, cax, texts, epoch)`: 
    - Uses the epoch number to fetch the corresponding confusion matrix values from 'all_data'.
    - Updates the matrix's color and text elements to reflect new values.
    - Returns the updated axis and color axis.

Note: The source of the matrix values is the 'data.csv' file, which should contain columns for 
True Positives, False Positives, False Negatives, and True Negatives for each epoch.

"""
all_data = pd.read_csv("data.csv")

#Function to create initial confusion matrix 
def create_initial_confusion_matrix():
    fig, ax = plt.subplots(figsize=(8, 6))

    vmin_value = min(all_data[['True Positives', 'False Positives', 'False Negatives', 'True Negatives']].min())
    vmax_value = max(all_data[['True Positives', 'False Positives', 'False Negatives', 'True Negatives']].max())
    cax = ax.matshow(np.zeros((2, 2)), cmap='Blues', vmin=vmin_value, vmax=vmax_value)
    cbar = plt.colorbar(cax)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.yaxis.offsetText.set_fontsize(8)

    ax.set_title('Confusion Matrix', fontsize=17, weight='bold')
    
    # Adjust x and y label sizes
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('True', fontsize=10)

    # Adjust tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=7)

    texts = [[None for _ in range(2)] for _ in range(2)]
    for i in range(2):
        for j in range(2):
            texts[i][j] = ax.text(j, i, '', ha='center', va='center', fontsize = 10)

    labels = ['Positive', 'Negative']
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

    return fig, ax, cax, texts

# Function to update the confusion matrix plot
def update_confusion_matrix(ax, cax, texts, epoch):
    data = [[all_data['True Positives'].iloc[epoch], all_data['False Positives'].iloc[epoch]],
            [all_data['False Negatives'].iloc[epoch], all_data['True Negatives'].iloc[epoch]]]
    cax.set_array(data)
    for i in range(2):
        for j in range(2):
            texts[i][j].set_text(str(data[i][j]))

    return ax, cax