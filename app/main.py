import os
import mimetypes
from flask import Flask, request, jsonify
import parselmouth

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def extract_parkinsons_features_parselmouth(file_path):
    try:
        sound = parselmouth.Sound(file_path)
        pitch = sound.to_pitch()

        # Jitter Measures
        jitter = pitch.get_jitter(method="local") * 100
        jitter_abs = pitch.get_jitter(method="local_absolute")
        jitter_rap = pitch.get_jitter(method="rap")
        jitter_ppq5 = pitch.get_jitter(method="ppq5")
        jitter_ddp = pitch.get_jitter(method="ddp")

        # Shimmer Measures
        shimmer = pitch.get_shimmer(method="local") * 100
        shimmer_db = pitch.get_shimmer(method="local_dB")
        shimmer_apq3 = pitch.get_shimmer(method="apq3")
        shimmer_apq5 = pitch.get_shimmer(method="apq5")
        shimmer_apq11 = pitch.get_shimmer(method="apq11")
        shimmer_dda = pitch.get_shimmer(method="dda")

        # Harmonics-to-Noise Ratio
        hnr = pitch.get_hnr()
        nhr = 1 / hnr if hnr != 0 else 0

        return {
            "Jitter(%)": jitter,
            "Jitter(Abs)": jitter_abs,
            "Jitter:RAP": jitter_rap,
            "Jitter:PPQ5": jitter_ppq5,
            "Jitter:DDP": jitter_ddp,
            "Shimmer": shimmer,
            "Shimmer(dB)": shimmer_db,
            "Shimmer:APQ3": shimmer_apq3,
            "Shimmer:APQ5": shimmer_apq5,
            "Shimmer:APQ11": shimmer_apq11,
            "Shimmer:DDA": shimmer_dda,
            "HNR": hnr,
            "NHR": nhr
        }
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

    features = extract_parkinsons_features_parselmouth(file_path)
    os.remove(file_path)

    return jsonify(features)


if __name__ == "__main__":
    app.run(debug=True)
