import os
import mimetypes
import subprocess

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


def get_audio_properties(file_path):
    """Check sample rate and number of channels using ffprobe."""
    command = [
        "ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
        "stream=sample_rate,channels", "-of", "csv=p=0", file_path
    ]
    try:
        output = subprocess.check_output(command).decode().strip()
        sample_rate, channels = map(int, output.split(","))
        return sample_rate, channels
    except Exception:
        return None, None  # Default in case of error


def resample_audio(input_path, output_path):
    """Resample audio to 16kHz mono if needed."""
    sample_rate, channels = get_audio_properties(input_path)

    # Skip resampling if already 16kHz mono
    if sample_rate == 16000 and channels == 1:
        return input_path  # Return the original file

    command = [
        "ffmpeg", "-i", input_path, "-ar", "16000", "-ac", "1", "-y", output_path
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return output_path if os.path.exists(output_path) else None


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
    """Extract Parkinson's features after resampling audio."""
    resampled_path = os.path.join(UPLOAD_FOLDER, "resampled_16k.wav")

    # Resample first
    processed_file = resample_audio(file_path, resampled_path)

    if not processed_file:
        return {"error": "Resampling failed"}

    try:
        # Extract features using OpenSMILE
        features = smile.process_file(processed_file)

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
            key: float(features[value].values[0].item()) for key, value in selected_features.items()
        }

        # Compute NHR as the inverse of HNR
        extracted_features["NHR"] = 1 / extracted_features["HNR"] if extracted_features["HNR"] != 0 else 0

        # Extract pitch data for nonlinear measures
        # Extract pitch data from LLD level
        # Extract pitch data for nonlinear measures
        pitch_data = smile_pitch.process_file(processed_file)

        if "F0final_sma" in pitch_data.columns and "voicingFinalUnclipped_sma" in pitch_data.columns:
            pitch_features = pitch_data[["F0final_sma", "voicingFinalUnclipped_sma"]].values.flatten()
        else:
            pitch_features = np.array([])

            # Compute RPDE, DFA, PPE
        extracted_features["RPDE"] = float(compute_rpde(pitch_features))
        extracted_features["DFA"] = float(compute_dfa(pitch_features))
        extracted_features["PPE"] = float(compute_ppe(pitch_features))

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
    if "error" in features:
        return jsonify(features), 400

    return jsonify({"features": features})  # Now structured and serializable


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
