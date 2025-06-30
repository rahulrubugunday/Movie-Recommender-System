# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('Dataset/netflix_titles.csv')
df

def create_textual_representation(row):
  textual_rep = f""" Type: {row['type']},
Title: {row['title']},
DIrector: {row['director']},
Cast: {row['cast']},
Released: {row['release_year']},
Genres: {row['listed_in']},
Description: {row['description']} """
  return textual_rep

df['textual_representation'] = df.apply(create_textual_representation, axis=1)
df['textual_representation'][0]

tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['textual_representation'])
tfidf_matrix

# Load a lightweight model for semantic embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for each movie's textual representation
embedding_matrix = model.encode(df['textual_representation'].tolist(), show_progress_bar=True)
embedding_matrix

dimension = embedding_matrix.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(np.array(embedding_matrix))
faiss_index

def hybrid_search(query, top_k=10, alpha=0.6):
    if not query.strip():
          return "I’m not a mind reader, just an underpaid algorithm. Give me something to work with"
    # Semantic embedding of user query
    query_embedding = model.encode([query])

    # Get top semantic results from FAISS
    D, I = faiss_index.search(np.array(query_embedding), top_k)
    semantic_scores = 1 / (1 + D[0])  # Convert L2 distances to similarity scores

    # Lexical (TF-IDF) similarity
    query_tfidf = tfidf.transform([query])
    lexical_scores_all = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    lexical_top_indices = lexical_scores_all.argsort()[::-1][:top_k]
    lexical_scores = lexical_scores_all[lexical_top_indices]

    # Merge the two sets of indices and scores
    score_dict = {}

    # Normalize scores
    sem_norm = (semantic_scores - np.min(semantic_scores)) / (np.max(semantic_scores) - np.min(semantic_scores) + 1e-8)
    lex_norm = (lexical_scores - np.min(lexical_scores)) / (np.max(lexical_scores) - np.min(lexical_scores) + 1e-8)

    for idx, score in zip(I[0], sem_norm):
        score_dict[idx] = alpha * score

    for idx, score in zip(lexical_top_indices, lex_norm):
        score_dict[idx] = score_dict.get(idx, 0) + (1 - alpha) * score

    # Sort by combined score
    top_indices = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # Display results
    for idx, score in top_indices:
        print(f"Title: {df.iloc[idx]['title']} \nYear: {df.iloc[idx]['release_year']}\nType: {df.iloc[idx]['type']}\nDescription: {df.iloc[idx]['description']}\nScore: {score:.4f}\n---")




#query = "Something fun like The Martian, but animated"
query = input("What are you looking for?\n")
hybrid_search(query, top_k=5)
