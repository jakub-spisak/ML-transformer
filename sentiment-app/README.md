# Transformer Sentiment App

A super‑small Streamlit interface around a fine‑tuned BERT.  
Add your `.pt` state_dict to `model/`, push to GitHub, and:

```bash
# local run
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
