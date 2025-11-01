import os
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI


RAIN_PATH = "data/Rainfall.csv"
CROP_PATH = "data/NewCrop.csv"

RAIN_EMB_FILE = "rain_embeds.npy"
RAIN_CTX_FILE = "rain_contexts.npy"

CROP_EMB_FILE = "crop_embeds.npy"
CROP_CTX_FILE = "crop_contexts.npy"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

openai_api_key = os.getenv("GROQ_API_KEY", "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")
client = OpenAI(api_key=openai_api_key, base_url="https://api.groq.com/openai/v1")


st.set_page_config(layout="wide", page_title="🌦️ Project Samarth – RAG Q&A on Govt Data")
st.title("Samarth – for Rainfall & Crop Data")
st.write(
    "Ask  questions "
)


@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

# Load datasets (without displaying tables)
try:
    rain_df = load_csv(RAIN_PATH)
except Exception as e:
    st.error(f"Couldn't load rainfall dataset: {e}")
    st.stop()

try:
    crop_df = load_csv(CROP_PATH)
except Exception as e:
    st.error(f"Couldn't load crop dataset: {e}")
    st.stop()

# Convert all text to lowercase for consistent search
rain_df = rain_df.applymap(lambda x: str(x).lower() if isinstance(x, str) else x)
crop_df = crop_df.applymap(lambda x: str(x).lower() if isinstance(x, str) else x)


@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)

embedder = load_embedder()

def make_contexts_rain(df):
    return [
        f"state: {r['State']}, district: {r['District']}, date: {r['Date']}, year: {r['Year']}, month: {r['Month']}, avg_rainfall: {r['Avg_rainfall']}, agency: {r['Agency_name']}"
        for _, r in df.iterrows()
    ]

def make_contexts_crop(df):
    return [
        f"year: {r['year']}, state: {r['state_name']}, district: {r['district_name']}, season: {r['season']}, crop: {r['crop_name']}, type: {r['crop_type']}, area: {r['area']}, production: {r['production']}"
        for _, r in df.iterrows()
    ]


def get_or_create_embeddings(ctx_file, emb_file, contexts, label):
    if os.path.exists(emb_file) and os.path.exists(ctx_file):
        st.info(f"✅ Loaded saved embeddings for {label}")
        ctx = np.load(ctx_file, allow_pickle=True).tolist()
        emb = np.load(emb_file)
    else:
        st.info(f"⚙️ Computing embeddings for {label} (this may take a while)...")
        progress = st.progress(0)
        emb = []
        batch_size = max(1, len(contexts)//100)
        for i in range(0, len(contexts), batch_size):
            batch = contexts[i:i+batch_size]
            emb.extend(embedder.encode(batch))
            progress.progress(min(1.0, i/len(contexts)))
        emb = np.array(emb)
        np.save(emb_file, emb)
        np.save(ctx_file, np.array(contexts, dtype=object))
        progress.progress(1.0)
        st.success(f"Embeddings for {label} created and cached.")
        ctx = contexts
    return ctx, emb


rain_contexts, rain_embeds = get_or_create_embeddings(RAIN_CTX_FILE, RAIN_EMB_FILE, make_contexts_rain(rain_df), "🌧️  Data")
crop_contexts, crop_embeds = get_or_create_embeddings(CROP_CTX_FILE, CROP_EMB_FILE, make_contexts_crop(crop_df), "🌾  Data")


st.markdown("---")
st.header("💬 Ask your question")

query = st.text_input("Type your question here:")
TOP_K = 5
SIM_THRESHOLD = 0.15

if query:
    query_emb = embedder.encode([query])
    sims_rain = cosine_similarity(query_emb, rain_embeds)[0]
    sims_crop = cosine_similarity(query_emb, crop_embeds)[0]

    top_rain_idx = sims_rain.argsort()[-TOP_K:][::-1]
    top_crop_idx = sims_crop.argsort()[-TOP_K:][::-1]

    top_rain = [(rain_contexts[i], sims_rain[i], f"rain-row-{i+2}") for i in top_rain_idx]
    top_crop = [(crop_contexts[i], sims_crop[i], f"crop-row-{i+2}") for i in top_crop_idx]

    combined_contexts = top_rain + top_crop
    combined_contexts = sorted(combined_contexts, key=lambda x: x[1], reverse=True)[:TOP_K]

    if combined_contexts[0][1] < SIM_THRESHOLD:
        st.warning("⚠️ Data not available in the provided datasets.")
    else:
        combined_text = "\n".join([f"[{ref}] {ctx}" for ctx, _, ref in combined_contexts])
        prompt = f"""
        You are a data assistant for the Government of India.
        Use ONLY the dataset information provided below (Rainfall + Crop) to answer the user's question.
        If the information is missing, respond: "Data not available in the provided datasets."
        Cite the row references you used.

        DATA:
        {combined_text}

        QUESTION:
        {query}

        ANSWER:
        """

        with st.spinner("🤖 Generating answer using  model..."):
            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=300,
            )
            answer = response.choices[0].message.content

        st.subheader("🧠 Model’s Answer:")
        st.write(answer)

        st.caption("📘 Sources (top matching rows from datasets):")
        st.dataframe(
            pd.DataFrame([
                {"Dataset": "Rainfall" if "rain" in ref else "Crop", "Ref": ref, "Similarity": f"{score:.3f}"}
                for _, score, ref in combined_contexts
            ])
        )
