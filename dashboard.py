import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd 

all_data = pd.read_csv("/Users/olivervu25/Documents/QUT/MXB362/data.csv")

def create_initial_plot():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(1, 30)
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

def simulate_training(progress_bar, plot_placeholder, total_epochs=30):
    fig, ax, line = create_initial_plot()
    for epoch in range(total_epochs):
        time.sleep(0.5)

        progress_bar.progress((epoch + 1) / total_epochs)  # Update the progress
        status_text.markdown(f"Epoch: <span style='color:red'>{epoch + 1}</span>", unsafe_allow_html=True)

        ax, line = update_plot(ax, line, epoch + 1)
        progress_bar.progress((epoch + 1) / total_epochs)
        plot_placeholder.pyplot(fig)  # This line updates the existing plot
        if epoch == 19: 
            ax.axvline(x=20, color='red', linestyle='--', lw=1)
    plt.close(fig)

st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)

st.title('Image Segmentation Model: Training Progress')

# Create columns for button and progress bar at the top
top_col1, top_col2 = st.columns([2, 8])

if top_col1.button('Start Training'):
    status_text = top_col2.empty()
    progress_bar = top_col2.progress(0)

    # Create columns for visualizations
    col1, col2 = st.columns([2, 3])
    with col1:
        st.header("Segmentation Training Plot")
        plot_placeholder = st.empty()
        simulate_training(progress_bar, plot_placeholder)  # Pass the plot_placeholder to the function

    with col2:
        st.header("Visualization 2")
        # ... other visualizations or content ...
