from ChernoffFace import *
import matplotlib.pyplot
import numpy
import matplotlib.cm
from constants import *
import matplotlib.pyplot as plt
import pandas as pd

def main():
    df, features, label_encoder = DF, FEATURES, LABEL_ENCODER

    features = features[:10]  # Use only the means and not the variances
    # Calculate average features for each genre
    average_features = variables_rescale(df[features].groupby('Genre').mean())
    average_features['Genre'] = label_encoder.classes_

    fig = chernoff_face(data=average_features,
                        n_columns=5,
                        long_face=True,
                        color_mapper=matplotlib.cm.tab20,
                        figsize=(4, 4), 
                        dpi=300)

    fig.tight_layout()
    matplotlib.pyplot.show()

if __name__ == '__main__':
    main()
