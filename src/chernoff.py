import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import numpy as np
from constants import *
from ChernoffFace import *
import matplotlib.cm as cm
from matplotlib.patches import PathPatch
from matplotlib.path import Path

# Mapping functions for facial features
def map_eye_size(size):
    min_size = 0.12
    max_size = 0.29
    return min_size + (max_size - min_size) * size

def map_eye_color(value):
    cmap = plt.get_cmap('rainbow')
    return cmap(value)

def map_nose_shape(value):
    # Function to create a triangle for the nose
    min_width = 0.01
    max_width = 0.2
    half_width = min_width + value * (max_width - min_width)
    return [(0.5 - half_width, 0.4), (0.5 + half_width, 0.4), (0.5, 0.6)]

def map_nose_color(value):
    cmap = plt.get_cmap('rainbow')
    return cmap(value)

def map_mouth_shape(value):    
    def _smile_curve(x, a):
        return a * x**2 + 0.3
    
    a = 3 * ((value - 0.5) * 2)
    X = np.linspace(-0.15, 0.15, 400)
    Y = _smile_curve(X, a)
    return X, Y

def map_mouth_color(value):
    cmap = plt.get_cmap('rainbow')
    return cmap(value)

def map_face_shape(value):
    """# Define face dimensions
    max_height = 0.9
    max_width = 0.8

    # Center of the face
    center_x = 0.5
    center_y = 0.5

    # Define vertices for the face outline
    verts = [
        (center_x, center_y - max_height / 2),  # mid of chin
        (center_x - 0.1, center_y - max_height / 2),
        (center_x - max_width / 2, center_y + max_height / 2 - 0.1), 
        (center_x, center_y + max_height / 2), 
        (center_x + max_width / 2, center_y + max_height / 2 - 0.1),
        (center_x + 0.1, center_y - max_height / 2),
        (center_x, center_y - max_height / 2),  # mid of chin
    ]

    # Define codes to create a closed path (MOVETO, LINETO, and CLOSEPOLY)
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]

    codes, verts = zip(*[
    (Path.MOVETO, (center_x, center_y - max_height / 2)),
    (Path.LINETO, (center_x - 0.1, center_y - max_height / 2)),
    (Path.CURVE4, (center_x - max_width / 2, center_y + max_height / 2 - 0.1)),
    (Path.CURVE4, (center_x, center_y + max_height / 2)),
    (Path.CURVE4, (center_x + max_width / 2, center_y + max_height / 2 - 0.1)),
    (Path.CURVE4, (center_x + 0.1, center_y - max_height / 2)),
    (Path.LINETO, (center_x, center_y - max_height / 2)),
    (Path.CLOSEPOLY, (center_x, center_y - max_height / 2))])
    
    face_path = patches.Path(verts, codes)
    
    return face_path"""
    min_width = 0.55
    max_width = 1.0
    
    min_height = 0.65
    max_height = 1.0
    
    face_shape = []
    
    face_shape.append(min_width + value * (max_width - min_width))
    face_shape.append(min_height + (1 - value) * (max_height - min_height))
    
    return face_shape

def map_face_color(value):
    cmap = plt.get_cmap('rainbow')
    return cmap(value)

# Function to create Chernoff face
def create_chernoff_face(parameters, ax, genre_name):
    eye_size = map_eye_size(parameters['eye_size'])
    nose_shape = map_nose_shape(parameters['nose_width'])
    nose_color = map_nose_color(parameters['nose_color'])
    eye_color = map_eye_color(parameters['eye_color'])
    x, y = map_mouth_shape(parameters['mouth_smile_factor'])
    mouth_color = map_mouth_color(parameters['mouth_color'])
    face_shape = map_face_shape(parameters['face_shape'])  
    face_color = map_face_color(parameters['face_color'])   
    
    # Draw face
    face = patches.Ellipse((0.5, 0.5), face_shape[0], face_shape[1], facecolor=face_color, edgecolor='black')

    #face_path = map_face_shape(parameters['face_shape'])
    #face = patches.PathPatch(face_path, edgecolor='black', facecolor='lightyellow')

    # Draw eyes
    eye_openness, eye_height, iris_size, pupil_size = 0.1, 0.65, 0.045, 0.025
    
    eye_left = patches.Ellipse((0.35, eye_height), eye_size, eye_openness, facecolor='white', edgecolor='black')
    iris_left = patches.Circle((0.35, eye_height), iris_size, color=eye_color)
    pupil_left = patches.Circle((0.35, eye_height), pupil_size, color='black')
    eye_right = patches.Ellipse((0.65, eye_height), eye_size, eye_openness, facecolor='white', edgecolor='black')
    iris_right = patches.Circle((0.65, eye_height), iris_size, color=eye_color)
    pupil_right = patches.Circle((0.65, eye_height), pupil_size, color='black')
    
    # Draw nose
    nose = patches.Polygon(nose_shape, facecolor=nose_color, closed=True, edgecolor='black')

    # Draw mouth
    mouth_edge = mlines.Line2D(x + 0.5, y, lw=7, color='black')
    mouth = mlines.Line2D(x + 0.5, y, lw=5, color=mouth_color)

    # Add facial features to the plot
    ax.add_patch(face)
    ax.add_patch(eye_left)
    ax.add_patch(eye_right)
    ax.add_patch(iris_left)
    ax.add_patch(iris_right)
    ax.add_patch(pupil_left)
    ax.add_patch(pupil_right)
    ax.add_patch(nose)
    ax.add_line(mouth_edge)
    ax.add_line(mouth)
    
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal')

    # Remove x and y ticks for cleaner visualization
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax.set_title(genre_name)
    
df, features, label_encoder = DF, FEATURES, LABEL_ENCODER

# features = features[:10]  # Use only the means and not the variances
# Calculate average features for each genre
average_features = variables_rescale(df[features].groupby('Genre').mean())
average_features['Genre'] = label_encoder.classes_

# print(average_features)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(15, 10), nrows=2, ncols=5)

chernoff_features = ['Chroma STFT μ', 'Chroma STFT σ²', 'RMS μ', 'RMS σ²', 'SC μ', 'SC σ²', 'Spectral BW μ', 'Spectral BW σ²']
genre_faces_vals = []

for i in range(10):  # For each genre
    genre_face_vals = []
    for chernoff_feature in chernoff_features:  # For each chosen feature for our Chernoff face
        genre_face_vals.append(average_features.at[i, chernoff_feature])  # Get the value for the feature
    genre_faces_vals.append(genre_face_vals)

for i in range(10):
    # Sample normalized parameters (replace with your actual parameters)
    parameters = {
        'eye_size': genre_faces_vals[i][0],             # Chroma STFT μ
        'eye_color': genre_faces_vals[i][1],            # Chroma STFT σ²
        'nose_width': genre_faces_vals[i][2],           # RMS μ
        'nose_color': genre_faces_vals[i][3],           # RMS σ²
        'mouth_smile_factor': genre_faces_vals[i][4],   # SC μ
        'mouth_color': genre_faces_vals[i][5],          # SC σ²
        'face_shape': genre_faces_vals[i][6],           # Spectral BW μ
        'face_color': genre_faces_vals[i][7],           # Spectral BW σ²
    }
    
    j = i // 5
    k = i % 5

    # Create and display the Chernoff face
    create_chernoff_face(parameters, ax[j, k], label_encoder.classes_[i])

plt.show()
