# --------------- IMPORT LIBRARIES ---------------
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from parcoords import load_data_from_file
import data

df, dimensions, _ = data.load_data_from_file('data/features_30_sec.csv')

# Group by genre and compute average values for each dimension
dimensions = [dim for dim in dimensions if dim['label'] != 'Genre']
grouped_df = df.groupby('Genre')[[dim['label'] for dim in dimensions]].mean().reset_index()

## Define different scales for each dimension
radial_axes = []
for dim in dimensions:
    axis_config = dict(
        range=[df[dim['label']].min(), df[dim['label']].max()],  # Set the range based on the data
        label=dim['label'],  # Use the dimension label as the axis label
    )
    radial_axes.append(axis_config)

# Create a radar chart with different scales
fig = go.Figure()

for i, row in grouped_df.iterrows():
    radar_values = [row[dim['label']] for dim in dimensions]
    radar_values.append(radar_values[0])  # Close the radar chart
    fig.add_trace(go.Scatterpolar(
        r=radar_values,
        theta=[dim['label'] for dim in dimensions],
        fill='toself',
        name=row['Genre'],  # Genre names as trace names
    ))

# Customize the radar chart layout with different radial axes
fig.update_layout(
    polar=dict(
        radialaxis=radial_axes,  # Use the defined radial axes
    ),
    showlegend=True,  # Set to False if you don't want a legend
    legend=dict(
        orientation='h',  # Horizontal legend
        x=0.5, y=1.15  # Positioning the legend
    )
)

fig.show()