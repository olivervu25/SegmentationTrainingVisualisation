import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd 
import streamlit as st

"""
Test Accuracy Visualization for Model Training

This script provides functions to visualize the test accuracy of an image segmentation model across epochs using Matplotlib. Additionally, it supports dynamic updates for visualization during the model's training process.

Primary Functions:
- `create_initial_plot()`: 
    - Initializes a line plot to depict the model's test accuracy over epochs.
    - Configures visual elements such as axis limits, labels, and title.
    - Returns the figure, axis, and line object, preparing it for updates.

- `update_plot(ax, line, current_epoch)`: 
    - Accepts the current epoch number to determine the segment of the data to display.
    - Fetches the accuracy values up to the current epoch from 'all_data' and updates the plot.
    - Returns the updated axis and line object.

Note: The source of the accuracy values is the 'data.csv' file, which should contain columns for Epoch and Segmentation Accuracy.

"""


all_data = pd.read_csv("data.csv")


def create_initial_plot():
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(1, 30)
    ax.set_ylim(min(all_data['Segmentation Accuracy']) - 0.01, max(all_data['Segmentation Accuracy']) + 0.01)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Segmentation Accuracy')
    ax.set_title('Test accuracy', fontsize=17, weight='bold')
    line, = ax.plot([], [], lw=2)
    return fig, ax, line
    
def update_plot(ax, line, current_epoch):
    x = all_data.Epoch[:current_epoch]
    y = all_data['Segmentation Accuracy'][:current_epoch]
    line.set_data(x, y)
    return ax, line
