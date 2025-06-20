import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load the trained models
model_pop = joblib.load('model_popularity.pkl')
model_stream = joblib.load('model_streams.pkl')

# App Title
st.title("🎵 Gwamz Spotify Track Success Predictor")
st.markdown("""
Welcome to the **Gwamz Song Success Predictor App**.  
Use the sidebar to set song features and predict the **Spotify Popularity Score** and **Estimated Streams**.
""")

# Sidebar Inputs
st.sidebar.header("📋 Input Song Features")

artist_followers = st.sidebar.number_input('Artist Followers', min_value=0, value=5000, step=500)
artist_popularity = st.sidebar.slider('Artist Popularity (0-100)', 0, 100, 60)
total_tracks_in_album = st.sidebar.slider('Total Tracks in Album', 1, 20, 1)
available_markets_count = st.sidebar.slider('Available Markets Count', 1, 100, 50)
track_number = st.sidebar.slider('Track Number in Album', 1, 20, 1)
disc_number = st.sidebar.slider('Disc Number', 1, 5, 1)
explicit = st.sidebar.selectbox('Explicit Content?', [0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')
track_age_days = st.sidebar.slider('Track Age (Days)', 0, 1000, 30)
release_day_of_week = st.sidebar.selectbox('Release Day of Week', [0,1,2,3,4,5,6],
                                           format_func=lambda x: ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'][x])
album_type_encoded = st.sidebar.selectbox('Album Type', [0,1,2],
                                          format_func=lambda x: ['Album','Single','Compilation'][x])
is_single = st.sidebar.selectbox('Is Single?', [0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')

input_features = np.array([[
    artist_followers,
    artist_popularity,
    total_tracks_in_album,
    available_markets_count,
    track_number,
    disc_number,
    explicit,
    track_age_days,
    release_day_of_week,
    album_type_encoded,
    is_single
]])

# Make predictions
predicted_popularity = model_pop.predict(input_features)[0]
predicted_streams = model_stream.predict(input_features)[0]

# Output Predictions
st.subheader("🎯 Predicted Song Performance")
st.write(f"**Spotify Popularity Score:** `{predicted_popularity:.2f}` / 100")
st.write(f"**Estimated Streams:** `{int(predicted_streams):,}` streams")

# Simulate Feature Importance (Placeholder Example)
feature_names = ['artist_followers', 'artist_popularity', 'total_tracks_in_album', 'available_markets_count',
                 'track_number', 'disc_number', 'explicit', 'track_age_days',
                 'release_day_of_week', 'album_type_encoded', 'is_single']

# You can replace these with model.feature_importances_ if available
importances = np.random.rand(len(feature_names))

importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

st.subheader("🔍 Feature Importance (Example)")
fig, ax = plt.subplots(figsize=(8,5))
ax.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
ax.invert_yaxis()
st.pyplot(fig)

st.markdown("---")
st.markdown("✅ **Model Performance Summary:**")
st.markdown("- Model Type: XGBoost Regressor")
st.markdown("- Evaluation: R²: 0.89, RMSE: 3.2 (Example)")  # Replace with your real metrics
st.markdown("- Data: Gwamz Spotify Historical Tracks (20 samples)")

# Footer
st.markdown("""
---
Made by **Your Data Science Team** · Powered by ML · Streamlit App
""")
