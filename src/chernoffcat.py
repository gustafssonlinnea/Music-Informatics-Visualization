import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.patches as patches
import matplotlib.lines as mlines
import numpy as np
from constants import *
from ChernoffFace import *
import matplotlib.cm as cm
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import math

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
    max_width = 0.1
    x_mid = 0.5
    y_mid = 0.4
    
    height = 0.05
    
    half_width = min_width + value * (max_width - min_width)
    vertices = [
                (x_mid - half_width, y_mid + height), 
                (x_mid + half_width, y_mid + height), 
                (x_mid, y_mid - height)
                ]
    return vertices

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
    min_width = 0.15
    max_width = 0.5
    
    min_height = 0.35
    max_height = 0.8
    
    face_shape = []
    
    # face_shape.append(min_width + value * (max_width - min_width))
    face_shape.append(0.5)
    face_shape.append(min_height + (1 - value) * (max_height - min_height))
    
    return face_shape

def map_face_color(value):
    cmap = plt.get_cmap('rainbow')
    return cmap(value)

def map_ear_shapes(value):
    # Function to create a triangle for the nose
    # min_width = 0.01
    # max_width = 0.2
    # half_width = min_width + value * (max_width - min_width)
    
    half_width = 0.15
    
    min_height = 0.08
    max_height = 0.23
    half_height = min_height + value * (max_height - min_height)
    
    mid_height = 0.89
    xpos_left = 0.21
    mid_left = (xpos_left, mid_height)
    mid_right = (1 - xpos_left, mid_height)
    
    theta = math.pi / 5
    
    ear_left = [
                (mid_left[0] - half_width, mid_left[1] - half_height), 
                (mid_left[0] + half_width, mid_left[1] - half_height), 
                (mid_left[0], mid_left[1] + half_height)
                ]
    
    rotated_ear_left = [
        ((x - mid_left[0]) * math.cos(theta) - (y - mid_left[1]) * math.sin(theta) + mid_left[0],
        (x - mid_left[0]) * math.sin(theta) + (y - mid_left[1]) * math.cos(theta) + mid_left[1])
        for x, y in ear_left
        ]
    
    ear_right = [
                (mid_right[0] - half_width, mid_right[1] - half_height), 
                (mid_right[0] + half_width, mid_right[1] - half_height), 
                (mid_right[0], mid_right[1] + half_height)
                ]
    
    rotated_ear_right = [
        ((x - mid_right[0]) * math.cos(-theta) - (y - mid_right[1]) * math.sin(-theta) + mid_right[0],
        (x - mid_right[0]) * math.sin(-theta) + (y - mid_right[1]) * math.cos(-theta) + mid_right[1])
        for x, y in ear_right
        ]
    
    return [rotated_ear_left, rotated_ear_right]

def map_whiskers(value_len, value_color, ax):
    num_whiskers = 3
    whisker_angle = math.pi / 4
    
    min_whisker_length = 0.2
    max_whisker_length = 0.5
    whisker_length = min_whisker_length + value_len * (max_whisker_length - min_whisker_length)
    
    cmap = plt.get_cmap('rainbow')
    whisker_color = cmap(value_color)
    
    left_center = (0.6, 0.4)
    right_center = (0.4, 0.4)
    
    left_angles = np.linspace(math.pi - whisker_angle / 2, math.pi + whisker_angle / 2, num_whiskers)
    right_angles = np.linspace(whisker_angle / 2, -whisker_angle / 2, num_whiskers)
    
    for angle_right, angle_left in zip(left_angles, right_angles):
        # whisker = np.linspace(0, whisker_length, 100)
        whisker = np.arange(0, whisker_length, 0.01)
        left_hair_x = [left_center[0] + i * np.cos(angle_left) for i in whisker]
        left_hair_y = [left_center[1] + i * np.sin(angle_left) for i in whisker]
        
        right_hair_x = [right_center[0] + i * np.cos(angle_right) for i in whisker]
        right_hair_y = [right_center[1] + i * np.sin(angle_right) for i in whisker]
            
        remove_first = 12
        ax.plot(left_hair_x[remove_first:], left_hair_y[remove_first:], color='black', linewidth=2)
        ax.plot(left_hair_x[remove_first:], left_hair_y[remove_first:], color=whisker_color)
        ax.plot(right_hair_x[remove_first:], right_hair_y[remove_first:], color='black', linewidth=2)
        ax.plot(right_hair_x[remove_first:], right_hair_y[remove_first:], color=whisker_color)
        
def map_whisker_color(value):
    cmap = plt.get_cmap('rainbow')
    return cmap(value)

# Function to create Chernoff face
def create_chernoff_face(parameters, ax, genre_name):
    face_shape = map_face_shape(parameters['face_shape'])  
    face_color = map_face_color(parameters['face_color'])
    # whisker_length = map_whiskers(parameters['whisker_length'])
    # whisker_color = map_whisker_color(parameters['whisker_color'])
    eye_size = map_eye_size(parameters['eye_size'])
    nose_shape = map_nose_shape(parameters['nose_width'])
    nose_color = map_nose_color(parameters['nose_color'])
    eye_color = map_eye_color(parameters['eye_color'])
    mouth_x, mouth_y = map_mouth_shape(parameters['mouth_smile_factor'])
    mouth_color = map_mouth_color(parameters['mouth_color'])
    ear_shapes = map_ear_shapes(parameters['ear_shape'])
    
    # Draw face
    face = patches.Ellipse((0.5, 0.6), 0.8, 0.7, facecolor=face_color)
    chin = patches.Ellipse((0.5, 0.4), face_shape[0], face_shape[1], facecolor=face_color)
    #face_path = map_face_shape(parameters['face_shape'])
    #face = patches.PathPatch(face_path, edgecolor='black', facecolor='lightyellow')
    
    # Draw ears
    ear_left = patches.Polygon(ear_shapes[0], facecolor=face_color, closed=True)
    ear_right = patches.Polygon(ear_shapes[1], facecolor=face_color, closed=True)

    # Draw eyes
    eye_openness, eye_height, iris_size, pupil_size = 0.1, 0.55, 0.045, 0.025
    eye_left_center = (0.35, eye_height)
    eye_right_center = (0.65, eye_height)
    # eye_angle_deg = -15
    
    eye_left = patches.Ellipse(eye_left_center, eye_size, eye_openness, facecolor='white', edgecolor='black')
    # Perform the rotation
    # transform = transforms.Affine2D().rotate_deg_around(eye_left_center[0], eye_left_center[1], eye_angle_deg)
    # eye_left = patches.Ellipse(eye_left_center, eye_size, eye_openness, facecolor='white', edgecolor='black', transform=transform)
    iris_left = patches.Circle(eye_left_center, iris_size, color=eye_color)
    pupil_left = patches.Circle(eye_left_center, pupil_size, color='black')
    
    eye_right = patches.Ellipse(eye_right_center, eye_size, eye_openness, facecolor='white', edgecolor='black')
    iris_right = patches.Circle(eye_right_center, iris_size, color=eye_color)
    pupil_right = patches.Circle(eye_right_center, pupil_size, color='black')
    
    # Draw nose
    nose = patches.Polygon(nose_shape, facecolor=nose_color, closed=True, edgecolor='black')

    # Draw mouth
    mouth_width = 5
    mouth_edge = mlines.Line2D(mouth_x + 0.5, mouth_y, lw=mouth_width + 2, color='black')
    mouth = mlines.Line2D(mouth_x + 0.5, mouth_y, lw=mouth_width, color=mouth_color)

    # Add facial features to the plot
    ax.add_patch(face)
    ax.add_patch(chin)
    map_whiskers(parameters['whisker_length'], parameters['whisker_color'], ax)
    ax.add_patch(ear_left)
    ax.add_patch(ear_right)
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
    ax.axis('off')  # Remove border around plot
    
    ax.set_title(genre_name)

def main(): 
    df, features, label_encoder = DF, FEATURES, LABEL_ENCODER

    # features = features[:10]  # Use only the means and not the variances
    # Calculate average features for each genre
    average_features = variables_rescale(df[features].groupby('Genre').mean())  # rescale to value [0, 1]
    average_features['Genre'] = label_encoder.classes_

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(15, 10), nrows=2, ncols=5)

    chernoff_features = ['Chroma STFT μ', 'Chroma STFT σ²', 
                        'RMS μ', 'RMS σ²', 
                        'SC μ', 'SC σ²', 
                        'Spectral BW μ', 'Spectral BW σ²',
                        'Spectral Rolloff μ', 'Spectral Rolloff σ²',
                        'Tempo']
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
            'whisker_length': genre_faces_vals[i][8],       # Spectral Rolloff μ
            'whisker_color': genre_faces_vals[i][9],        # Spectral Rolloff σ²
            'ear_shape': genre_faces_vals[i][10],           # Tempo
        }

        # Create and display the Chernoff face
        create_chernoff_face(parameters, 
                            ax[i // 5, i % 5], 
                            label_encoder.classes_[i])
        
    # plt.savefig('images/chernoff-genre-cats.png', dpi=300)  # Set dpi to 300
    plt.show()

if __name__ == '__main__':
    main()