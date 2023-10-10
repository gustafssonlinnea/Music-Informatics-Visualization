import pandas as pd
import plotly.express
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from constants import *

def main():
    # --------------- LOAD AND PREPARE DATA ---------------
    df, dimensions, features, label_encoder = DF, DIMENSIONS, FEATURES, LABEL_ENCODER
    X = df[features]

    # --------------- APPLY T-SNE ---------------
    # Initialize t-SNE with desired perplexity and other parameters
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)

    # Apply t-SNE to your data
    X_tsne = tsne.fit_transform(X)

    # --------------- VISUALIZE USING GO ---------------
    # Define the trace for the scatter plot
    scatter_trace = go.Scatter(
        x=X_tsne[:, 0],
        y=X_tsne[:, 1],
        mode='markers',
        marker=dict(
            size=12,
            opacity=0.7,
            color=df['Genre'],
            colorscale=COLORS,
            colorbar=dict(
                title='Genre',
                tickvals=list(range(len(label_encoder.classes_))),
                ticktext=label_encoder.classes_,
                tickmode='array',  # Use tickvals and ticktext as array values for the axis ticks
            ),
            showscale=True,
            cmin=0,
            cmax=len(label_encoder.classes_) - 1,
        ),
        text=label_encoder.inverse_transform(df['Genre']),  # Set the text (hover information) to the Genre column values
    )

    # Define the layout for the scatter plot
    layout = go.Layout(
        title='t-SNE Visualization of Audio Features in GTZAN Dataset',
        title_x=0.5,  # Set x to 0.5 for center alignment horizontally
        # title_y=0.95,  # Set y to 0.9 for near the top alignment vertically
        xaxis=dict(title='t-SNE Dimension 1'),
        yaxis=dict(title='t-SNE Dimension 2'),
    )

    # Create the figure and update the layout and data
    fig = go.Figure(data=[scatter_trace], layout=layout)
    # Show the figure
    fig.show()

if __name__ == '__main__':
    main()
