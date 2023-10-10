# --------------- IMPORT LIBRARIES ---------------
import plotly.graph_objects as go
# import plotly.express as px
from constants import *

def main():   
    # --------------- LOAD DATA ---------------         
    df, dimensions, label_encoder = DF, DIMENSIONS, LABEL_ENCODER

    # --------------- CREATE PLOT ---------------
    # Create Parcoords trace
    parcoords_trace = go.Parcoords(
        line=dict(color=df['Genre'], colorscale=COLORS),
        dimensions=dimensions
    )

    # Create the figure and add the Parcoords trace
    fig = go.Figure(data=parcoords_trace)
    
    # Create a scatter plot for the color bar with hidden tick labels and white background
    colorbar_trace = go.Scatter(
        x=[1, 1],
        y=[1, 1],
        marker=dict(
            opacity=0,
            colorscale=COLORS,
            showscale=True,
            cmin=0,
            cmax=len(label_encoder.classes_) - 1,
            colorbar=dict(
                title='Genre',
                tickvals=list(range(len(label_encoder.classes_))),
                ticktext=label_encoder.classes_,
                tickmode='array',  # Use tickvals and ticktext as array values for the axis ticks
            )
        ),
        showlegend=False,
        line=dict(width=0),  # Hide the border line of the scatter marker
        hoverinfo='none'  # Hide hover information
    )

    # Add the color bar trace to the figure
    fig.add_trace(colorbar_trace)

    # Remove x- and y-axis ticks
    fig.update_layout(
        xaxis = dict(
            tickmode = 'array',
            tickvals = [],
            ticktext = []
        ),
        yaxis = dict(
            tickmode = 'array',
            tickvals = [],
            ticktext = []
        )
    )

    # Update the layout
    fig.update_layout(
        title='Parallel Coordinates Plot for Audio Features of GTZAN Dataset',  # Add a title
        margin=dict(t=150),  # Increase top margin to provide space for the title
        title_x=0.5,  # Set x to 0.5 for center alignment horizontally
        title_y=0.95,  # Set y to 0.9 for near the top alignment vertically
        plot_bgcolor='rgba(0,0,0,0)',  # Make the plot background transparent
        # paper_bgcolor='rgba(0,0,0,0)'  # Make the paper background transparent
    )

    # Show figure
    fig.show()
    
if __name__ == '__main__':
    main()


"""
# Create a parallel coordinates plot using the numerical label encoding
fig = px.parallel_coordinates(
    subset_df,
    dimensions=dimensions,
    color="Genre",
    color_continuous_scale=px.colors.qualitative.Plotly  # Set your desired colorscale
)"""


"""
# Set custom axis ticks and labels for the 'Genre' dimension
genre_labels = label_encoder.classes_  # Get the original genre labels
genre_values = label_encoder.transform(genre_labels)  # Get the corresponding encoded values

fig = go.Figure(data=
    go.Parcoords(
        line = dict(color = df['Genre'],  # Assuming 'Genre_encoded' is the column representing genre
                   colorscale = px.colors.qualitative.Plotly),  # You can specify any colorscale you prefer
        dimensions = [dict(range = [df[dim].min(), df[dim].max()],
                           label = dim, values = df[dim]) for dim in dimensions]
    )
)
"""