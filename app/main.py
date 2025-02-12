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
        selected_features = [
            "jitterLocal_sma_amean",
            "jitterDDP_sma_amean",
            "jitterLocal_sma_stddev",
            "jitterDDP_sma_stddev",
            "shimmerLocal_sma_amean",
            "shimmerLocal_sma_stddev",
            "logHNR_sma_amean",
            "logHNR_sma_stddev"
        ]
        features = smile.process_file(file_path)
        extracted_features = features[selected_features].values[0]
        print(extracted_features)

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

    return jsonify(features)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Render's assigned port
    app.run(host="0.0.0.0", port=port)
