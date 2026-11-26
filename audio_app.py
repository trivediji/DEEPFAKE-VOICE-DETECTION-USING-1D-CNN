"""Tkinter GUI for the baseline CNN deepfake voice classifier."""
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import joblib
import librosa
import numpy as np
from tensorflow.keras.models import load_model
from pydub import AudioSegment
import simpleaudio as sa

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "cnn_deepfake_audio_model.h5"
SCALER_PATH = BASE_DIR / "scaler.pkl"

# Load the model and scaler saved by the baseline notebook.
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

if getattr(scaler, "n_features_in_", 26) != 26:
    raise ValueError("The loaded scaler must be fitted on 26 features.")

root = tk.Tk()
root.title("Deepfake Voice Detector")
root.geometry("440x360")
root.configure(bg="#1e1e1e")
root.resizable(False, False)

file_path = None


def extract_features(audio_file):
    """Extract the 26 features used by the project's training pipeline."""
    y, sr = librosa.load(audio_file, sr=22050, duration=4)
    y = librosa.util.fix_length(y, size=sr * 4)

    features = [
        librosa.feature.chroma_stft(y=y, sr=sr).mean(),
        librosa.feature.rms(y=y).mean(),
        librosa.feature.spectral_centroid(y=y, sr=sr).mean(),
        librosa.feature.spectral_bandwidth(y=y, sr=sr).mean(),
        librosa.feature.spectral_rolloff(y=y, sr=sr).mean(),
        librosa.feature.zero_crossing_rate(y=y).mean(),
    ]
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    features.extend(mfccs.mean(axis=1))

    features = np.asarray(features, dtype=np.float64)
    if features.size != 26:
        raise ValueError(f"Expected 26 features, but extracted {features.size}.")
    return features.reshape(1, 26)


def select_audio():
    global file_path
    selected = filedialog.askopenfilename(
        title="Select an audio file",
        filetypes=[("WAV audio", "*.wav")],
    )
    if selected:
        file_path = selected
        file_label.config(text=Path(selected).name)
        detect_button.config(state=tk.NORMAL)
        play_button.config(state=tk.NORMAL)
        result_var.set("Ready to analyze")


def detect_audio():
    if not file_path:
        messagebox.showwarning("No audio", "Please select a WAV file first.")
        return
    try:
        features = extract_features(file_path)
        scaled = scaler.transform(features)

        # Conv1D expects (batch, features, channels), not (batch, features).
        cnn_input = scaled.reshape(1, 26, 1)
        prediction = float(model.predict(cnn_input, verbose=0)[0][0])
        label = "REAL" if prediction >= 0.5 else "FAKE"
        result_var.set(f"Prediction score: {prediction:.4f}\nClassification: {label}")
        result_label.config(fg="#5ee6a8" if label == "REAL" else "#ff7373")
    except Exception as exc:
        messagebox.showerror("Detection error", f"Could not analyze audio:\n{exc}")


def play_audio():
    if not file_path:
        messagebox.showwarning("No audio", "Please select a WAV file first.")
        return
    try:
        audio = AudioSegment.from_wav(file_path)
        playback = sa.play_buffer(
            audio.raw_data,
            num_channels=audio.channels,
            bytes_per_sample=audio.sample_width,
            sample_rate=audio.frame_rate,
        )
        playback.wait_done()
    except Exception as exc:
        messagebox.showerror("Playback error", f"Could not play audio:\n{exc}")


# GUI
heading = tk.Label(
    root, text="Deepfake Voice Detection", font=("Helvetica", 18, "bold"),
    bg="#1e1e1e", fg="white",
)
heading.pack(pady=(22, 14))

file_label = tk.Label(
    root, text="No audio file selected", font=("Helvetica", 11),
    bg="#1e1e1e", fg="#bbbbbb", wraplength=380,
)
file_label.pack(pady=8)

select_button = tk.Button(root, text="Choose WAV Audio", command=select_audio,
                         width=20, bg="#444444", fg="white")
select_button.pack(pady=10)

play_button = tk.Button(root, text="Play Audio", command=play_audio,
                        width=20, bg="#2468a8", fg="white", state=tk.DISABLED)
play_button.pack(pady=6)

detect_button = tk.Button(root, text="Detect Voice", command=detect_audio,
                          width=20, bg="#23864a", fg="white", state=tk.DISABLED)
detect_button.pack(pady=6)

result_var = tk.StringVar(value="Select an audio file to begin")
result_label = tk.Label(root, textvariable=result_var, font=("Helvetica", 12, "bold"),
                        bg="#1e1e1e", fg="#62d8e8", pady=14)
result_label.pack(pady=10)

root.mainloop()
