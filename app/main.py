import os
import mimetypes
from flask import Flask, request, jsonify
import opensmile
import pandas as pd

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize OpenSMILE with Parkinson’s-related features
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
)

def extract_parkinsons_features(file_path):
    try:
        # Extract features using OpenSMILE
        features = smile.process_file(file_path)
        features_dict = features.iloc[0].to_dict()

        # Filter only relevant Parkinson’s features
        required_features = {
            "Jitter(%)": features_dict.get("jitterLocal_sma"),
            "Jitter(Abs)": features_dict.get("jitterDDP_sma"),
            "Jitter:RAP": features_dict.get("jitterRap_sma"),
            "Jitter:PPQ5": features_dict.get("jitterPpq5_sma"),
            "Shimmer": features_dict.get("shimmerLocal_sma"),
            "Shimmer(dB)": features_dict.get("shimmerLocalDb_sma"),
            "Shimmer:APQ3": features_dict.get("shimmerApq3_sma"),
            "Shimmer:APQ5": features_dict.get("shimmerApq5_sma"),
            "Shimmer:APQ11": features_dict.get("shimmerApq11_sma"),
            "HNR": features_dict.get("HNRdB_sma")
        }

        return required_features
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

    return jsonify(features)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Render's assigned port
    app.run(host="0.0.0.0", port=port)
