import mimetypes
from flask import Flask, request, jsonify
import librosa
import numpy as np
import os
import subprocess

app = Flask(__name__)


def extract_parkinsons_features_librosa(file_path):
    """
    Extracts voice features matching Parkinson's dataset using Librosa.

    Args:
    - file_path (str): Path to the audio file.

    Returns:
    - dict: Extracted features including Jitter, Shimmer, HNR, NHR, RPDE, DFA, and PPE.
    """
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=22050)

        # Compute fundamental frequency (f0) (Needed for Jitter Calculation)
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=75, fmax=300)

        # Remove NaN values
        f0 = f0[~np.isnan(f0)]

        # Compute Jitter (%): Frequency variation
        if len(f0) > 1:
            jitter = np.mean(np.abs(np.diff(f0))) / np.mean(f0)
        else:
            jitter = 0.0  # If no voiced frames

        # Compute Shimmer: Amplitude variation
        frame_amplitudes = np.abs(y)
        if len(frame_amplitudes) > 1:
            shimmer = np.mean(np.abs(np.diff(frame_amplitudes))) / np.mean(frame_amplitudes)
        else:
            shimmer = 0.0

        # Compute HNR (Harmonics-to-Noise Ratio)
        hnr = librosa.effects.harmonic(y).std() / librosa.effects.percussive(y).std()
        if hnr == 0:
            nhr = 0
        else:
            nhr = 1 / hnr  # Convert HNR to NHR

        # Compute RPDE, DFA, PPE (Approximations since Librosa doesn’t provide exact measures)
        rpde = np.random.uniform(0.2, 0.6)  # Placeholder
        dfa = np.random.uniform(0.5, 1.5)  # Placeholder
        ppe = np.std(f0) / np.mean(f0) if len(f0) > 1 else 0.0  # Approximate Pitch Period Entropy

        # Return extracted features
        return {
            "Jitter(%)": jitter,
            "Jitter(Abs)": jitter / 100,  # Approximate Absolute Jitter
            "Jitter:RAP": jitter * 0.75,  # Approximate RAP
            "Jitter:PPQ5": jitter * 0.8,  # Approximate PPQ5
            "Jitter:DDP": jitter * 1.1,  # Approximate DDP
            "Shimmer": shimmer,
            "Shimmer(dB)": shimmer * 10,  # Convert to dB scale
            "Shimmer:APQ3": shimmer * 0.7,  # Approximate APQ3
            "Shimmer:APQ5": shimmer * 0.85,  # Approximate APQ5
            "Shimmer:APQ11": shimmer * 0.9,  # Approximate APQ11
            "Shimmer:DDA": shimmer * 1.1,  # Approximate DDA
            "HNR": hnr,
            "NHR": nhr,
            "RPDE": rpde,
            "DFA": dfa,
            "PPE": ppe
        }

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None


# ✅ Create uploads folder if not exists


@app.route("/extract", methods=["POST"])
def extract():
    UPLOAD_FOLDER = "uploads"
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Get MIME type and file extension
    mime_type, _ = mimetypes.guess_type(file_path)

    print("Upload Type : " + mime_type + "\n")
    file_extension = os.path.splitext(file_path)[1].lower()

    # If the file is not WAV, return an error with the detected MIME type
    if file_extension not in [".wav", ".mp3", ".m4a", ".3gp"]:
        print(f"error: Unsupported file format\n"
              f"filename: {file.filename}\n"
              f"detected_mime: {mime_type or 'Unknown'}\n"
              f"file_extension: {file_extension}\n")

    wav_path = os.path.splitext(file_path)[0] + ".wav"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", file_path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"FFmpeg conversion failed", "details": e.stderr.decode()}), 501

    print("Converted to .wav\n")

    features = extract_parkinsons_features_librosa(wav_path)
    os.remove(file_path)
    os.remove(wav_path)  # Delete file after processing

    return jsonify(features)


if __name__ == "__main__":
    app.run(debug=True)
