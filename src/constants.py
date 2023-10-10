import plotly.express as px
import data

FILEPATH = 'data/features_30_sec.csv'
DF, DIMENSIONS, FEATURES, LABEL_ENCODER = data.load_data_from_file(FILEPATH)
COLORS = px.colors.qualitative.Plotly
