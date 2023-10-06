import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd 
from Accuracy import create_initial_plot, update_plot
from Confusion_matrix import create_initial_confusion_matrix, update_confusion_matrix
from Mask import frame_return, load_frames
import os

all_data = pd.read_csv("/Users/olivervu25/Documents/QUT/MXB362/project/data.csv")

if os.path.exists("/Users/olivervu25/Documents/QUT/MXB362/project/frames.pkl"):
    frames = load_frames("/Users/olivervu25/Documents/QUT/MXB362/project/frames.pkl")
else:
    frames = frame_return()  # This will generate and save frames to frames.pkl
    
def simulate_training(progress_bar, matrix_placeholder, plot_placeholder, gif_placeholder, total_epochs=30):
    # Display the GIF using the gif_placeholder

    fig1, ax1, line = create_initial_plot()
    fig2, ax2, cax, texts = create_initial_confusion_matrix()

    for epoch in range(total_epochs):
        time.sleep(0.2)  # Simulate some time delay for each epoch
        
        # Update the line plot
        ax1, line = update_plot(ax1, line, epoch + 1)
        plot_placeholder.pyplot(fig1)
        
        # Update the confusion matrix
        ax2, cax = update_confusion_matrix(ax2, cax, texts, epoch)
        matrix_placeholder.pyplot(fig2)

        # Display the corresponding frame for the GIF
        gif_placeholder.image(frames[epoch], caption=f'Predicted Masks Epoch: {epoch+1}')

        
        progress_bar.progress((epoch + 1) / total_epochs)  # Update the progress
        status_text.markdown(f"Epoch: <span style='color:red'>{epoch + 1}</span>", unsafe_allow_html=True)
        
        if epoch == 19: 
            ax1.axvline(x=20, color='red', linestyle='--', lw=1)
            plot_placeholder.pyplot(fig1)  # Re-render the plot with the vertical line

    plt.close(fig1)
    plt.close(fig2)


st.markdown("""
        <style>
               .block-container {
                    padding-top: 0rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)

st.title('Image Segmentation Model: Training Progress')

# Create columns for button and progress bar at the top
top_col1, top_col2 = st.columns([2, 3])

if top_col1.button('Start Training'):
    status_text = top_col2.empty()
    progress_bar = top_col2.progress(0)

    # Create columns for visualizations
    col1, col2 = st.columns([2, 5])
    
    matrix_placeholder = col1.empty()  # For confusion matrix
    plot_placeholder = col1.empty()  # For line plot
    gif_placeholder = col2.empty()  # Placeholder for the GIF

    simulate_training(progress_bar, matrix_placeholder, plot_placeholder, gif_placeholder)
