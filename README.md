#  Project Samarth – Intelligent Q&A over Government Datasets

This project is a **Streamlit-based AI Q&A system** that allows users to ask natural language questions about Indian government datasets — specifically **Rainfall** and **Crop Production** — and get intelligent answers using **Groq (OpenAI-compatible API)**.

---

##  Features
- Interactive Q&A interface powered by **Streamlit**
- Uses **Sentence Transformers** for semantic search in CSV data
- Integrates **Rainfall** and **Crop datasets**
- Smart retrieval + LLM reasoning via **Groq API**
- Auto-saves and reuses embeddings for fast startup

---

##  Tech Stack
- Python, Streamlit, Pandas, NumPy  
- Sentence Transformers (`all-MiniLM-L6-v2`)  
- Groq LLM API (OpenAI-compatible interface)

---

##  Environment Setup

Create a .env file and add your API key:

XAI_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx

#  Data Files

Place your CSV files under the data/ folder:

data/
├── Rainfall.csv
          └── Crop.csv

##  Run the App
streamlit run app.py


##  Installation

```bash
git clone <repo-link>
cd Project_Samarth
pip install -r requirements.txt
```
## Example Questions

“What is the average rainfall in Kurung Kumey, Arunachal Pradesh for July 2024?”

“Compare rainfall between Kurung Kumey and Nanded districts.”
