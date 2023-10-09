# --------------- IMPORT LIBRARIES ---------------
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder


def load_data_from_file(filepath: str, num_songs: int=None):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(filepath)

    # Use label encoding to convert categorical labels to numerical values
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['label'])

    # Custom names for each column
    custom_dimension_names = {
        'label_encoded': 'Genre',
        'tempo': 'Tempo',
        'chroma_stft_mean': 'Chroma STFT μ',
        'chroma_stft_var': 'Chroma STFT σ²',
        'rms_mean': 'RMS μ',
        'rms_var': 'RMS σ²',
        'spectral_centroid_mean': 'SC μ',
        'spectral_centroid_var': 'SC σ²',
        'spectral_bandwidth_mean': 'Spectral BW μ',
        'spectral_bandwidth_var': 'Spectral BW σ²',
        'rolloff_mean': 'Spectral Rolloff μ',
        'rolloff_var': 'Spectral Rolloff σ²',
        'zero_crossing_rate_mean': 'ZC Rate μ',
        'zero_crossing_rate_var': 'ZC Rate σ²',
        'harmony_mean': 'Harmony μ',
        'harmony_var': 'Harmony σ²',
        # 'perceptr_mean': 'Perceptual μ',
        # 'perceptr_var': 'Perceptual σ²',
    }

    df.rename(columns=custom_dimension_names, inplace=True)

    # Define the indices for excluding MFCC columns (from 19th to 59th column)
    mfcc_start_index = 19
    mfcc_end_index = 59

    # Exclude 'filename', etc, and MFCC columns from dimensions
    dimensions_to_exclude = ['filename', 'Genre', 'length', 'perceptr_mean', 'perceptr_var'] + list(df.columns[mfcc_start_index:mfcc_end_index+1])
    dimensions = [col for col in df.columns if col not in dimensions_to_exclude]

    # Randomly sample n rows from the DataFrame
    if num_songs is not None:
        df = df.sample(n=num_songs, random_state=42)  # Set random_state for reproducibility

    # Customize the axis ticks for the 'Genre' dimension
    genre_labels = label_encoder.classes_  # Get the original genre labels
    genre_values = label_encoder.transform(genre_labels)  # Get the corresponding encoded values

    # Define a mapping dictionary for Genre encoding to Genre names
    genre_mapping = {val: label for val, label in zip(genre_values, genre_labels)}

    # Create dimensions for Parcoords
    dimensions = []

    # Add 'Genre' as the first dimension
    genre_dimension = dict(
        range=[df['Genre'].min(), df['Genre'].max()],
        label='Genre',
        values=df['Genre'],
        tickvals=list(genre_mapping.keys()),
        ticktext=list(genre_mapping.values()),
        tickformat=".0f"
    )
    
    tempo_dimension = dict(
        range=[df['Tempo'].min(), df['Tempo'].max()],
        label='Tempo',
        values=df['Tempo']
    )
    
    #dimensions.append(genre_dimension)
    # dimensions.append(tempo_dimension)

    for label in custom_dimension_names.values():
        tickvals = list(genre_mapping.keys()) if label == 'Genre' else None
        ticktext = list(genre_mapping.values()) if label == 'Genre' else None
        dimension = dict(range=[df[label].min(), df[label].max()],
                        label=label,
                        values=df[label],
                        tickvals=tickvals,
                        ticktext=ticktext,
                        tickformat=".0f" if label == 'Genre' else None)
        #if label != 'Genre' or 'Tempo': 
        dimensions.append(dimension)     
            
    return df, dimensions, label_encoder

def main():   
    # --------------- LOAD DATA ---------------         
    df, dimensions, label_encoder = load_data_from_file(
        filepath='data/features_30_sec.csv')

    # --------------- CREATE PLOT ---------------
    # Create Parcoords trace
    parcoords_trace = go.Parcoords(
        line=dict(color=df['Genre'], colorscale=px.colors.qualitative.Plotly),
        dimensions=dimensions
    )

    # Create the figure and add the Parcoords trace
    fig = go.Figure(data=parcoords_trace)
    
    # Create a scatter plot for the color bar with hidden tick labels and white background
    colorbar_trace = go.Scatter(
        x=[1, 1],
        y=[1, 1],
        marker=dict(
            colorscale=px.colors.qualitative.Plotly,
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
        title='Parallel Coordinates Plot for GTZAN Dataset',  # Add a title
        margin=dict(t=150),  # Increase top margin to provide space for the title
        title_x=0.5,  # Set x to 0.5 for center alignment horizontally
        title_y=0.95,  # Set y to 0.9 for near the top alignment vertically
        plot_bgcolor='rgba(0,0,0,0)',  # Make the plot background transparent
        # paper_bgcolor='rgba(0,0,0,0)'  # Make the paper background transparent
    )

    # Show figure
    fig.show()


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