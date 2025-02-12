import os
import mimetypes
from flask import Flask, request, jsonify
import opensmile
import scipy.stats
import nolds
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize OpenSMILE with Parkinson’s-related features
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
)

smile_pitch = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,  # Corrected!
)

def compute_rpde(signal):
    """ Compute Recurrence Period Density Entropy (RPDE) """
    signal = signal[signal > 0]  # Remove unvoiced parts
    if len(signal) < 2:
        return 0  # Return default if data is insufficient
    return scipy.stats.entropy(np.histogram(signal, bins=10)[0])


def compute_dfa(signal):
    """ Compute Detrended Fluctuation Analysis (DFA) """
    if len(signal) < 10:
        return 0  # Default if data is insufficient
    return nolds.dfa(signal)


def compute_ppe(signal):
    """ Compute Pitch Period Entropy (PPE) """
    signal = signal[signal > 0]  # Remove unvoiced parts
    if len(signal) < 2:
        return 0  # Default if data is insufficient
    return scipy.stats.entropy(signal)


def extract_parkinsons_features(file_path):
    try:
        # Extract features using OpenSMILE
        features = smile.process_file(file_path)

        # Selected feature mapping
        selected_features = {
            "Jitter(%)": "jitterLocal_sma_amean",
            "Jitter:PPQ5": "jitterLocal_sma_rqmean",
            "Jitter:DDP": "jitterLocal_sma_amean",
            "Shimmer": "shimmerLocal_sma_amean",
            "Shimmer(dB)": "shimmerLocal_sma_amean",
            "Shimmer:APQ3": "shimmerLocal_sma_quartile1",
            "Shimmer:APQ5": "shimmerLocal_sma_quartile2",
            "Shimmer:APQ11": "shimmerLocal_sma_quartile3",
            "Shimmer:DDA": "shimmerLocal_sma_de_amean",
            "NHR": "logHNR_sma_amean",  # Inverse of HNR
            "HNR": "logHNR_sma_amean",
        }

        extracted_features = {
            key: float(features[value].values[0]) for key, value in selected_features.items()
        }

        # Extract pitch data for nonlinear measures
        # Extract pitch data from LLD level
        pitch_features = smile_pitch.process_file(file_path)[
            ["F0final_sma", "voicingFinalUnclipped_sma"]].values.flatten()

        # Compute RPDE, DFA, PPE
        extracted_features["RPDE"] = compute_rpde(pitch_features)
        extracted_features["DFA"] = compute_dfa(pitch_features)
        extracted_features["PPE"] = compute_ppe(pitch_features)

        return extracted_features

    except Exception as e:
        return {"error": f"Processing error: {e}"}


@app.route("/extract", methods=["POST"])
def extract():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Ensure correct file format
    mime_type, _ = mimetypes.guess_type(file_path)
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension not in [".wav", ".mp3", ".m4a", ".3gp"]:
        os.remove(file_path)
        return jsonify({"error": f"Unsupported format: {mime_type or 'Unknown'} ({file_extension})"}), 400

    features = extract_parkinsons_features(file_path)
    os.remove(file_path)

    # Handle errors from feature extraction
    if isinstance(features, dict):
        return jsonify(features), 400

    return jsonify({"features": features})  # Now structured and serializable


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
