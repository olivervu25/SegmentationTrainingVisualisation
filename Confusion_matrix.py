import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd

#This file is to create confusion matrix

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