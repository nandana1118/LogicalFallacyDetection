# Logical Fallacy Detection System

An AI-based system for detecting and explaining logical fallacies in text using a fine-tuned BERT model.

## Features

- Detects 13 types of logical fallacies
- Accepts direct text input or article URLs
- Provides confidence score for predictions
- Uses LIME for explainable AI
- Generates human-readable explanations
- Interactive Streamlit web interface

## Technologies Used

- Python
- BERT (Transformers)
- PyTorch
- Streamlit
- LIME
- Scikit-learn
- BeautifulSoup

## Project Structure

- `data/` - Dataset, preprocessing, augmentation
- `model/` - Training and evaluation scripts
- `app/` - Main application modules
- `notebooks/` - Colab experiments

## How to Run

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py