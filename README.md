# DeepFake Voice Detection Using CNN

A machine-learning project that analyzes audio features and uses a Convolutional Neural Network (CNN) to classify a voice recording as **REAL** or **FAKE**.

## Overview

This project explores deepfake voice detection using audio feature extraction, feature scaling, and a CNN classifier. A desktop GUI allows users to select an audio file and view the model's prediction.

## Workflow

1. Select an audio file through the desktop application.
2. Extract a set of 26 audio features.
3. Apply the saved feature scaler.
4. Reshape the feature vector to match the CNN input format `(1, 26, 1)`.
5. Run inference using the trained CNN.
6. Display the classification result in the GUI.

## Technologies

- Python
- TensorFlow / Keras
- NumPy
- Librosa
- Joblib
- Pydub
- Simpleaudio
- Tkinter

## Project files

```text
DeepFake-Voice-Detection/
├── audio_app_fixed.py
├── cnn_deepfake_audio_model.h5
├── scaler.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/DeepFake-Voice-Detection.git
cd DeepFake-Voice-Detection
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. (Recommended) Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows**
```bash
.venv\Scripts\activate
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Run the application

```bash
python audio_app_fixed.py
```

Use the application's interface to select a supported audio file and run the prediction.

## Model and preprocessing

The application uses a saved CNN model and a saved scaler. The feature vector contains 26 features, and the scaled input is reshaped to `(1, 26, 1)` before inference.

**Important:** The model and scaler must come from the same training/preprocessing pipeline. Do not substitute other model or scaler files without checking compatibility.

## Limitations

- Predictions are model outputs, not proof that a recording is authentic or manipulated.
- Performance depends on the training data, preprocessing, audio quality, and the types of voice manipulation encountered.
- The model's performance on unseen speakers, languages, recording conditions, or new synthesis methods has not been established here.
- This repository should not be used as the sole basis for high-stakes decisions.

## Future improvements

- Evaluate the model on a held-out dataset and report relevant metrics.
- Add more robust audio-format handling and validation.
- Improve the GUI and add confidence/score interpretation.
- Test generalization across speakers, languages, and manipulation methods.

## License

A license has not yet been selected. Add a license file before granting reuse permissions.
