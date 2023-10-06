import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd

all_data = pd.read_csv("/Users/olivervu25/Documents/QUT/MXB362/project/data.csv")

# # Set up the figure, the axis, and the plot elements
# fig, ax = plt.subplots(figsize=(6, 6))

# vmin_value = min(all_data[['True Positives', 'False Positives', 'False Negatives', 'True Negatives']].min())
# vmax_value = max(all_data[['True Positives', 'False Positives', 'False Negatives', 'True Negatives']].max())

# cax = ax.matshow(np.zeros((2, 2)), cmap='Blues', vmin=vmin_value, vmax=vmax_value)
# plt.colorbar(cax)
# texts = [[None for _ in range(2)] for _ in range(2)]

# # For the text labels
# for i in range(2):
#     for j in range(2):
#         texts[i][j] = ax.text(j, i, '', ha='center', va='center')

# # Set up the x and y axis labels
# labels = ['Positive', 'Negative']
# ax.set_xticks([0, 1])
# ax.set_yticks([0, 1])
# ax.set_xticklabels(labels)
# ax.set_yticklabels(labels)
# ax.set_xlabel('Predicted')
# ax.set_ylabel('True')
# ax.set_title('Confusion Matrix - Epoch 1')


# # Initialization function: Plot the background of each frame
# def init():
#     for text_row in texts:
#         for text in text_row:
#             text.set_text('')
#     return cax

# # Animation function: This will be called sequentially
# def animate(epoch):
#     data = [[all_data['True Positives'].iloc[epoch], all_data['False Positives'].iloc[epoch]],
#             [all_data['False Negatives'].iloc[epoch], all_data['True Negatives'].iloc[epoch]]]
#     cax.set_array(data)
#     ax.set_title(f'Confusion Matrix - Epoch {epoch + 1}')
    
#     # Set data values
#     for i in range(2):
#         for j in range(2):
#             texts[i][j].set_text(str(data[i][j]))

#     return cax,

# # Call the animator
# ani = FuncAnimation(fig, animate, frames=30, init_func=init, interval=1000)  # 1000 ms = 1 second

# plt.tight_layout()
# plt.show()

# # saves the animation
# ani.save('confusion_matrix_animation.gif', writer='imagemagick', fps=1)


def create_initial_confusion_matrix():
    fig, ax = plt.subplots(figsize=(8, 6))

    vmin_value = min(all_data[['True Positives', 'False Positives', 'False Negatives', 'True Negatives']].min())
    vmax_value = max(all_data[['True Positives', 'False Positives', 'False Negatives', 'True Negatives']].max())
    cax = ax.matshow(np.zeros((2, 2)), cmap='Blues', vmin=vmin_value, vmax=vmax_value)
    cbar = plt.colorbar(cax)
    cbar.ax.tick_params(labelsize=3)
    cbar.ax.yaxis.offsetText.set_fontsize(3)

    ax.set_title('Confusion Matrix', fontsize=20)
    
    # Adjust x and y label sizes
    ax.set_xlabel('Predicted', fontsize=3)
    ax.set_ylabel('True', fontsize=3)

    # Adjust tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=3)


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