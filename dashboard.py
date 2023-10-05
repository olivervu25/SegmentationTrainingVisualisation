import streamlit as st
import time

def simulate_training(progress_bar, total_epochs=30):
    for epoch in range(total_epochs):
        time.sleep(1)  # Simulate some time delay for each epoch
        progress_bar.progress((epoch + 1) / total_epochs)  # Update the progress
        status_text.markdown(f"Epoch: <span style='color:red'>{epoch + 1}</span>", unsafe_allow_html=True)

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

# Create columns for button and progress bar
col1, col2 = st.columns([2, 8])  # Adjust the numbers to modify the relative width of the columns

# Place the button in the first column
if col1.button('Start Training'):
    # Add a label and place the progress bar in the second column
    status_text = col2.empty()  # Placeholder for the text
    progress_bar = col2.progress(0)
    simulate_training(progress_bar)
