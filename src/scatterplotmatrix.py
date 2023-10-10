import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import data

df, dimensions, features, _ = data.load_data_from_file()

# Assuming df contains your data and 'Genre' column represents the categories
# Generate a list of colors based on unique genres in your DataFrame
unique_genres = df['Genre'].unique()
num_genres = len(unique_genres)
colors = plt.cm.rainbow(np.linspace(0, 1, num_genres))

# Map genre labels to color indices
genre_to_color = {genre: color for genre, color in zip(unique_genres, colors)}
color_indices = df['Genre'].map(genre_to_color)
"""
# Plot the scatter matrix using the generated color indices
pd.plotting.scatter_matrix(df[features], figsize=(20, 20), grid=True, marker='o')#, c=list(color_indices))
plt.show()
"""

"""
# Define colors for each genre
genre_colors = {
    'blues': 'blue',
    'classical': 'red',
    'country': 'yellow',
    'disco': 'purple',
    'hip-hop': 'orange',
    'jazz': 'cyan',
    'metal': 'black',
    'pop': 'pink',
    'reggae': 'green',
    'rock': 'grey'
}

# Map genre labels to colors
colors = df['Genre'].map(genre_colors)"""


pd.plotting.scatter_matrix(df[features], figsize=(20,20),grid=True,
                           marker='o')#, c=df['Genre'].map(colors))


# Create scatterplot matrix with histograms on the diagonal using Seaborn
#sns.set(style="ticks")
#sns.pairplot(df[columns], diag_kind="kde", hue='Genre', markers='o')
plt.show()
