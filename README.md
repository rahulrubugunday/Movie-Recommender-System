🎬 ChatGPT-Style Hybrid Movie Recommendation System

This is a lightweight, hybrid movie recommendation engine that interprets **natural language prompts** to suggest relevant movies. Inspired by the conversational flexibility of ChatGPT and the personalization of Netflix, this system combines **semantic embeddings** and **TF-IDF** to deliver curated movie suggestions.


## 📌 Features

- 💬 Natural language input – “Something like Interstellar, but lighter and romantic”
- 🧠 Semantic understanding via SentenceTransformers
- 📚 TF-IDF lexical search for keyword-based matching
- ⚖️ Hybrid scoring algorithm combining both
- ⚡ Fast vector search using FAISS

## 📥 Dataset

- Source: [Netflix Movies and TV Shows – Kaggle](https://www.kaggle.com/datasets/shivamb/netflix-shows)
- Place `netflix_titles.csv` in the `dataset/` folder before running.

---

## 🚀 How It Works

1. **Textual Representation**: Each movie is converted into a string with title, cast, director, genre, and description.
2. **Vectorization**:
   - `SentenceTransformers` generates semantic embeddings.
   - `TF-IDF` vectorizer captures keyword-level features.
3. **Hybrid Search**:
   - FAISS finds the closest matches in embedding space.
   - Cosine similarity over TF-IDF vectors is computed.
   - Final score:  
     `score = alpha * semantic_similarity + (1 - alpha) * lexical_similarity`

---

## 🧪 How to Use

Run the recommendation engine from the `src/` directory:

```bash
cd src
python main.py
```

You'll be prompted to enter a natural language query like:

```
What are you looking for?
> I want a dark crime drama like Narcos but with more politics
```

The script will return the top recommended titles with metadata and hybrid scores.

---

## 📦 Dependencies

To run this project, install the following Python libraries:

```bash
pip install pandas numpy scikit-learn faiss-cpu sentence-transformers
```

You can also install them one at a time:

- `pandas`
- `numpy`
- `scikit-learn`
- `faiss-cpu`
- `sentence-transformers`

This project runs on Python 3.7 or later.

---

## 🧠 Example Output

```
Title: The Social Dilemma
Year: 2020
Type: Movie
Description: Tech experts sound the alarm on the dangerous human impact of social networking.
Score: 0.8741
---
```

---

## 👨‍💻 Author

**Rahul Rubugunday**  
Graduate Student | Full-Stack Engineer | AI/ML Enthusiast  
📧 [rrahul97@gmail.com](mailto:rrahul97@gmail.com)

---
