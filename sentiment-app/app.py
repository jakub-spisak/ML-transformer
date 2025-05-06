import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------- config ----------
MODEL_PATH = "model/sentiment_model.pt"
BASE_MODEL  = "bert-base-uncased"          # change if you fine‑tuned another backbone
LABELS      = ["negative", "positive"]     # index 0 → neg, 1 → pos
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
# -----------------------------

@st.cache_resource(show_spinner=False)
def load_model():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    net = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(LABELS)
    )
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    net.load_state_dict(state)
    net.to(DEVICE).eval()
    return tok, net

tokenizer, model = load_model()

def classify(text: str) -> dict[str, float]:
    """Return class probabilities."""
    encoded = tokenizer(
        text,
        truncation=True,
        padding=True,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        out = model(**encoded).logits.softmax(dim=-1)[0]

    return {lbl: float(out[i]) for i, lbl in enumerate(LABELS)}

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Sentiment classifier", layout="centered")

st.title("📝 Transformer Sentiment Demo")
txt = st.text_area("Paste some text", height=200, placeholder="Type here…")

if st.button("Analyze", disabled=not txt.strip()):
    with st.spinner("Crunching numbers…"):
        probs = classify(txt)
    st.success(
        f"**{max(probs, key=probs.get).upper()}** "
        f"({probs['positive']:.2%} positive · {probs['negative']:.2%} negative)"
    )
    st.bar_chart(probs)
