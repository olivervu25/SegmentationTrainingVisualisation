import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd 
import streamlit as st


all_data = pd.read_csv("/Users/olivervu25/Documents/QUT/MXB362/data.csv")

# Setting up the plot
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.set_xlim(1, 30)  # Assuming you have 30 epochs
# ax.set_ylim(min(all_data['Segmentation Accuracy']) - 0.01, max(all_data['Segmentation Accuracy']) + 0.01)
# ax.set_xlabel('Epoch')
# ax.set_ylabel('Segmentation Accuracy')
# line, = ax.plot([], [], lw=2)
# vline_added = False  # Flag to indicate if the vertical line is added

# # Initialization function: Plot the background of each frame
# def init():
#     line.set_data([], [])
#     return line,

# # Animation function: This will be called sequentially
# def animate(i):
#     x = all_data.Epoch[:i]
#     y = all_data['Segmentation Accuracy'][:i]
#     line.set_data(x, y)
#     if i == 20: 
#         ax.axvline(x=20, color='red', linestyle='--', lw=1)

#     return line,

# # Call the animator
# ani = FuncAnimation(fig, animate, frames=len(all_data), init_func=init, blit=False, interval=500)  # 500 ms = 0.5 second

# plt.show()


# Function to create an initial plot
def create_initial_plot():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(1, 30)  # Assuming you have 30 epochs
    ax.set_ylim(min(all_data['Segmentation Accuracy']) - 0.01, max(all_data['Segmentation Accuracy']) + 0.01)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Segmentation Accuracy')
    
    line, = ax.plot([], [], lw=2)
    return fig, ax, line

def update_plot(ax, line, current_epoch):
    x = all_data.Epoch[:current_epoch]
    y = all_data['Segmentation Accuracy'][:current_epoch]
    line.set_data(x, y)
    return ax, line