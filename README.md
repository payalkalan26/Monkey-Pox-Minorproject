# Monkeypox Detection Demo

This project provides two ways to present your work:

1. Streamlit interactive demo (`app.py`) for image uploads and live prediction.
2. Static showcase site under `site/` for a polished write-up with visuals.

## Project Structure

```
monkeypox-demo/
├─ app.py                # Streamlit app
├─ requirements.txt      # Python dependencies (add your model framework)
├─ config.json           # Runtime config (create this, see example below)
├─ assets/               # Plots, sample images, figures for the app
├─ models/               # Place your saved model file here
├─ site/                 # Static showcase website
│  ├─ index.html
│  ├─ style.css
│  └─ assets/
└─ README.md
```

## Configure Your Model

Create a `config.json` at the project root with your settings. Example:

```json
{
  "framework": "tensorflow",
  "model_path": "models/monkeypox_cnn.h5",
  "input_size": [224, 224],
  "to_rgb": true,
  "normalize": true,
  "mean": [0.485, 0.456, 0.406],
  "std": [0.229, 0.224, 0.225],
  "labels": ["Monkeypox", "Non-Monkeypox"]
}
```

Supported frameworks out-of-the-box: `sklearn`, `keras`/`tensorflow`, `pytorch`, `onnx`.
Install the matching package in your environment and add it to `requirements.txt`.

## Run the Interactive App (Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# If needed, also: pip install tensorflow  (or torch / scikit-learn / onnxruntime)
streamlit run app.py
```

Then open the local URL shown by Streamlit.

## Add Your Assets

- Put model file under `models/`, update `model_path` in `config.json`.
- Put plots and images under `assets/` to display in the Results tab.
- Edit text in `app.py` sections to add your description.

## Static Showcase Site

- Files in `site/` are plain HTML/CSS for easy hosting.
- Open `site/index.html` in a browser or deploy to Netlify/GitHub Pages.
- Replace placeholder texts and image references with your content.

## Notes

- This demo is for educational purposes and not for clinical use.
- Ensure you have the right to use and share images and data.
