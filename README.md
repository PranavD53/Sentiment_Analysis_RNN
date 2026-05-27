# 🧠 MindSense — AI-Based Mental Health Sentiment Monitoring System

> Emotion Detection using Simple Recurrent Neural Networks · Built with Streamlit

---

## Overview

MindSense is a web-based mental health sentiment monitoring application that analyzes free-form text input and classifies the underlying emotion using a trained Recurrent Neural Network (RNN). It provides real-time predictions, confidence scores, emotion probability distributions, and personalized emotional guidance — all within a minimalistic, luxury-styled interface.

---

## Features

- **Emotion Classification** — Detects emotional states such as happiness, anxiety, sadness, and neutrality from user-entered text
- **Confidence Scoring** — Displays the model's certainty as a percentage alongside the predicted label
- **Probability Chart** — Visual bar chart showing the full distribution across all emotion classes
- **Sample Sentences** — One-click sample injection that populates the input and auto-runs analysis
- **Emotional Guidance** — Context-aware wellness tips and motivational messages based on the predicted emotion
- **Dark Luxury UI** — Navy and gold design with serif/sans-serif typography, smooth interactions, and responsive layout

---

## Project Structure

```
project/
│
├── app.py                          # Main Streamlit application
│
├── sentiment_model_files/
│   ├── rnn_model.h5                # Trained Keras RNN model
│   ├── tokenizer.pkl               # Fitted Keras tokenizer
│   └── label_encoder.pkl           # Fitted sklearn LabelEncoder
│
└── README.md
```

---

## Requirements

### Python Version
Python 3.8 or higher is recommended.

### Dependencies

Install all required packages using pip:

```bash
pip install streamlit tensorflow numpy matplotlib nltk scikit-learn
```

Or create a `requirements.txt`:

```
streamlit>=1.32.0
tensorflow>=2.12.0
numpy>=1.23.0
matplotlib>=3.7.0
nltk>=3.8.0
scikit-learn>=1.2.0
```

Then install with:

```bash
pip install -r requirements.txt
```

### NLTK Data

The app downloads required NLTK data automatically on first run:

```python
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")
```

---

## Setup & Run

**1. Clone or download the repository**

```bash
git clone https://github.com/your-username/mindsense.git
cd mindsense
```

**2. Ensure model files are in place**

Place your trained model artifacts inside a folder named `sentiment_model_files/`:

```
sentiment_model_files/
├── rnn_model.h5
├── tokenizer.pkl
└── label_encoder.pkl
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Launch the app**

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

---

## How It Works

### Text Preprocessing Pipeline

Raw user input is cleaned through the following steps before being passed to the model:

1. Converted to lowercase
2. Punctuation removed
3. Digits stripped
4. Tokenized using NLTK's `word_tokenize`
5. English stopwords filtered out
6. Rejoined as a clean string

### Prediction Pipeline

```
Raw Text
   └─► clean_text()
         └─► tokenizer.texts_to_sequences()
               └─► pad_sequences(maxlen=100)
                     └─► model.predict()
                           └─► argmax → label + confidence
```

The model outputs a probability vector over all emotion classes. The class with the highest probability is selected as the predicted emotion, and its probability value is the confidence score.

---

## UI Sections

| Section | Description |
|---|---|
| **Header** | App title, subtitle, and project identity |
| **About** | Explains emotional AI, NLP, and RNN's role in sequence learning |
| **Text Input** | Multi-line textarea for free-form text entry |
| **Sample Selector** | Dropdown with pre-written example sentences; auto-fills and analyzes on click |
| **Prediction Output** | Three metric cards showing emotion, confidence %, and emotional status |
| **Probability Chart** | Dark-themed bar chart with the predicted class highlighted in gold |
| **Emotional Guidance** | Color-coded card with three actionable wellness tips based on the detected emotion |

---

## Emotional Guidance Logic

| Predicted Emotion | Guidance Theme | Card Color |
|---|---|---|
| Happy / Positive / Joyful | Sustain positivity, share energy, anchor the moment | Green |
| Sad / Anxious / Stressed | Breathing exercises, connection, grounding | Amber |
| Neutral / Other | Reflection, gentle movement, intentional planning | Blue |

---

## Model Details

| Attribute | Value |
|---|---|
| Architecture | Simple RNN (Recurrent Neural Network) |
| Input length | 100 tokens (padded/truncated) |
| Preprocessing | NLTK tokenization + stopword removal |
| Serialization | Keras `.h5` + Python `pickle` |

> The model, tokenizer, and label encoder must be trained and saved separately before running this app. The app loads them at startup using `@st.cache_resource` for efficiency.

---

## Deployment

### Local
Run `streamlit run app.py` as described above.

### Streamlit Community Cloud

1. Push your project to a public GitHub repository
2. Ensure `requirements.txt` is present at the root
3. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repository
4. Set the main file path to `app.py`
5. Add your model files or load them from a remote source (e.g. Google Drive, Hugging Face Hub) since large binary files are not recommended in Git

> **Note:** `rnn_model.h5` may be large. Consider using [Git LFS](https://git-lfs.github.com/) or hosting the model externally and downloading it at runtime.

---

## Known Limitations

- The model's accuracy depends entirely on the quality and diversity of training data
- Short or ambiguous inputs (e.g. "ok", "fine") may yield low-confidence predictions
- The app does not store or log any user input — all processing is in-memory only
- Not intended as a substitute for professional mental health support

---

## License

This project is developed for academic and educational purposes.

---

## Author

Developed as part of an AI/ML academic project on emotion detection and mental health monitoring using Natural Language Processing and Deep Learning.