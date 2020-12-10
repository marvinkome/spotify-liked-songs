import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

minmaxScaler = MinMaxScaler()
model = pickle.load(open("classification_model.sav", 'rb'))
liked_songs = pd.read_json("./data/spotify-liked-songs.json")

feat_cols = [
    'valence', 'acousticness', 'danceability', 'energy', 'instrumentalness',
    'liveness', 'loudness', 'speechiness'
]
df_liked = liked_songs[feat_cols]

loudness = df_liked["loudness"].values
df_liked["loudness"] = minmaxScaler.fit_transform(loudness.reshape(-1, 1))

liked_songs['cluster'] = model.predict(df_liked)
liked_songs.to_json("./output/liked_songs_clustered.json", orient='records')
