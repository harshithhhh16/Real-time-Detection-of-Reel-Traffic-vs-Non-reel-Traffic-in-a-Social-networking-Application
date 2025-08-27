# Technical Documentation

## approach to solve this problem and what makes it unique

SNS applications (such as Facebook, YouTube, Instagram) transmit both video (short videos, reels, stories) and non-video traffic (feeds, text posts, comments, suggestions) through the same data pipeline. This project aims to build an AI model to differentiate reel/video traffic from non-reel/video traffic in real-time. This enables user equipment (UE) to dynamically optimize network performance, streaming quality, and data usage based on traffic type.

What makes this solution unique is its ability to distinguish video vs non-video traffic based on network traffic features alone, without decrypting payload data, which respects user privacy while maintaining accuracy under varying network conditions including congestion and coverage fluctuations.

---

## Technical Stack

- **Programming Language:** Python 3.9+
- **Web Framework:** [Streamlit](https://streamlit.io/) (for real-time UI and interaction)
- **Machine Learning:** [scikit-learn](https://scikit-learn.org/) (model training and inference)
- **Data Processing:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Model Persistence:** [Joblib](https://joblib.readthedocs.io/)
- **Visualization:** [Plotly](https://plotly.com/python/) (charts and gauges)
- **Version Control & Hosting:** [GitHub](https://github.com/), local deployment or cloud Streamlit deployment

---

## The Technical Architecture of Your Solution

1. **Data Collection:** Network traffic flow data collected from SNS app usage, labeled as video or non-video.

2. **Feature Extraction:** Extracts flow-level traffic features such as packet counts, byte sizes, packet inter-arrival times, and packet length statistics.

3. **Model Training:** Trains ML classifiers (e.g., Random Forest, SVM) on labeled traffic features to classify video vs non-video flows.

4. **Model Serialization:** Trained models and associated preprocessing (scalers, feature lists) saved as `.pkl` files.

5. **Real-time Classification UI:** Streamlit app loads selected model version, accepts real-time or simulated network feature input, performs classification, and visualizes prediction with confidence.

6. **Quick Tests:** Preloaded sample network patterns (e.g., typical video streaming vs web browsing) to demonstrate model predictions instantly.

7. **Results Dashboard:** Displays prediction probabilities, confidence indicators (gauge chart), and detailed bar charts for transparency.

---

## Implementation Details

- The Streamlit app dynamically loads available model versions from the `src/models` folder.
- Users select model versions by timestamp.
- Input features for network flow stats are entered manually or tested with pre-defined patterns.
- The app scales input data with the saved scaler, performs prediction, and shows results with confidence metrics.
- Predictions distinguish between `STREAMING` (video/reel traffic) and `NON-STREAMING` (non-video traffic).
- Visualization uses Plotly's indicator gauges and bar charts integrated inside Streamlit.

---

## Installation Instructions

1. Clone the repository and navigate into the `src/` folder:
  
2. Install required Python packages:
3. Ensure the `models/` folder contains your trained model files:
- `video_classifier_TIMESTAMP.pkl`
- `scaler_TIMESTAMP.pkl`
- `features_TIMESTAMP.pkl`
- `metadata_TIMESTAMP.pkl`

4. Run the Streamlit app:


5. Open your browser and navigate to: [http://localhost:8501](http://localhost:8501/)

---

## User Guide

- **Model Selection:** Use the sidebar to select the trained model version by timestamp.
- **Input Features:** Manually enter network traffic features or use pre-set quick test buttons (Video or Web patterns).
- **Classify Traffic:** Click "Classify Traffic" to get the prediction.
- **Results:** View predicted traffic type with confidence percentage, probability breakdown, and visual gauges.
- **Quick Tests:** Instant classification for typical patterns to understand model behavior.

---

## Salient Features of the Project

- Real-time classification of SNS network traffic into video vs non-video categories.
- User-friendly interface with detailed prediction insights.
- Multiple model versions support with automatic loading.
- Robust handling of traffic under variable network conditions.
- Privacy-preserving by using flow metadata rather than payload inspection.
- Open-source and extensible, enabling easy integration into real-world network monitoring tools.





