import pandas as pd
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
        'rms_mean': 'RMS μ',
        'spectral_centroid_mean': 'SC μ',
        'spectral_bandwidth_mean': 'Spectral BW μ',
        'rolloff_mean': 'Spectral Rolloff μ',
        'zero_crossing_rate_mean': 'ZC Rate μ',
        'perceptr_mean': 'Perceptrual μ',
        'harmony_mean': 'Harmony μ',
        'chroma_stft_var': 'Chroma STFT σ²',
        'rms_var': 'RMS σ²',
        'spectral_centroid_var': 'SC σ²',
        'spectral_bandwidth_var': 'Spectral BW σ²',
        'rolloff_var': 'Spectral Rolloff σ²',
        'zero_crossing_rate_var': 'ZC Rate σ²',
        'perceptr_var': 'Perceptrual σ²',
        'harmony_var': 'Harmony σ²',
    }
    features = list(custom_dimension_names.values())
    
    df.rename(columns=custom_dimension_names, inplace=True)

    # Define the indices for excluding MFCC columns (from 19th to 59th column)
    mfcc_start_index = 19
    mfcc_end_index = 59

    """# Exclude 'filename', etc, and MFCC columns from dimensions
    dimensions_to_exclude = ['filename', 'Genre', 'length', 'perceptr_mean', 'perceptr_var'] \
        + list(df.columns[mfcc_start_index:mfcc_end_index + 1])
    dimensions = [col for col in df.columns if col not in dimensions_to_exclude]"""

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

    for label in features:
        tickvals = list(genre_mapping.keys()) if label == 'Genre' else None
        ticktext = list(genre_mapping.values()) if label == 'Genre' else None
        dimension = dict(range=[df[label].min(), df[label].max()],
                        label=label,
                        values=df[label],
                        tickvals=tickvals,
                        ticktext=ticktext,
                        tickformat=".0f" if label == 'Genre' else None)
        dimensions.append(dimension)     
            
    return df, dimensions, features, label_encoder