import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd 
from Accuracy import create_initial_plot, update_plot
from Confusion_matrix import create_initial_confusion_matrix, update_confusion_matrix
from Mask import frame_return, load_frames
import os

all_data = pd.read_csv("data.csv")

# Global state for the training status

class GlobalState:
    def __init__(self):
        self.training_complete = False

global_state = GlobalState()

if os.path.exists("frames2.pkl"):
    frames = load_frames("frames2.pkl")
else:
    frames = frame_return()  # This will generate and save frames to frames.pkl
    
def simulate_training(progress_bar, matrix_placeholder, plot_placeholder, gif_placeholder, total_epochs=30):
    # Display the GIF using the gif_placeholder

    fig1, ax1, line = create_initial_plot()
    fig2, ax2, cax, texts = create_initial_confusion_matrix()

    for epoch in range(total_epochs):
        
        # Update the line plot
        ax1, line = update_plot(ax1, line, epoch + 1)
        plot_placeholder.pyplot(fig1)
        
        # Update the confusion matrix
        ax2, cax = update_confusion_matrix(ax2, cax, texts, epoch)
        matrix_placeholder.pyplot(fig2)

        # Display the corresponding frame for the GIF
        gif_placeholder.image(frames[epoch])
        
        if epoch == 19: 
            ax1.axvline(x=20, color='red', linestyle='--', lw=1, label='MobileNetV3Small Unfrozen')
            ax1.legend(loc='upper left')  # Display the legend on the upper right corner
            plot_placeholder.pyplot(fig1)  # Re-render the plot with the vertical line and legend
        
        progress_bar.progress((epoch + 1) / total_epochs)  # Update the progress
        if epoch < 19:  # remember Python is 0-indexed
            status_text.markdown(f"Epoch: <span style='color:red'>{epoch + 1}</span>", unsafe_allow_html=True)
        else:
            status_text.markdown(f"Epoch: <span style='color:red'>{epoch + 1}</span> - <span style='font-weight: bold'>MobileNetV3Small Unfrozen</span>", unsafe_allow_html=True)

        space.markdown(f" ", unsafe_allow_html=True)
        space1.markdown(f" ", unsafe_allow_html=True)
        space2.markdown(f" ", unsafe_allow_html=True)

        time.sleep(0.2)  # Simulate some time delay for each epoch


    plt.close(fig1)
    plt.close(fig2)


st.markdown("""
        <style>
                .stProgress > div > div {
                    background-color: blue !important;
                }
               .block-container {
                    padding-top: 0rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }

        </style>
        """, unsafe_allow_html=True)

st.title('Image Segmentation Model: Training Progress', help='Click to start or restart the training process')
st.markdown('<span style="font-size:14px">The model used is a pretrained MobileNetV3Small tailored for segmentation on the Oxford Pets dataset. It\'s trained for 30 epochs, with the first 20 epochs keeping the MobileNetV3Small frozen.</span>', unsafe_allow_html=True)

# Create columns for visualizations
col1, col2 = st.columns([2, 5])

# Within the main right column (col2), create additional columns for the button and progress bar
top_col1, top_col2, top_col3 = col2.columns([1, 6, 1])

if global_state.training_complete:
    button_text = 'Restart Training'
else:
    button_text = 'Start Training'

space = top_col2.empty()
if top_col2.button('Start Training'):
    
    # Resetting state if it's a restart
    if global_state.training_complete:
        global_state.training_complete = False
        progress_bar = top_col2.empty()
        space1 = top_col2.empty()
        space2 = top_col2.empty()
        status_text = top_col2.empty()
        matrix_placeholder = col1.empty()
        plot_placeholder = col1.empty()
        gif_placeholder = col2.empty()


    status_text = top_col2.empty()
    progress_bar = top_col2.progress(0)
    space1 = top_col2.empty()
    space2 = top_col2.empty()
    matrix_placeholder = col1.empty()  # For confusion matrix
    plot_placeholder = col1.empty()  # For line plot
    gif_placeholder = col2.empty()  # Placeholder for the GIF

    simulate_training(progress_bar, matrix_placeholder, plot_placeholder, gif_placeholder)

    # Once the training is completed
    global_state.training_complete = True
    status_text.markdown("<span style='color:green'>Training Completed!</span>", unsafe_allow_html=True)

# After the training is completed, try changing to the red styling
st.markdown("""
    <style>
        .stProgress > div > div {
            background-color: red !important;
        }
    </style>
    """, unsafe_allow_html=True)









