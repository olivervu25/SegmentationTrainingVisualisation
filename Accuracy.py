import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd 
import streamlit as st


all_data = pd.read_csv("/Users/olivervu25/Documents/QUT/MXB362/project/data.csv")


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
