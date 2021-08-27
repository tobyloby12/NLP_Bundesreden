import pandas as pd
import numpy as np
from summarisation_bert import *
import os
from pdf_file_extraction import *
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


model_de = spacy.load('de_core_news_sm')
german_stopwords = stopwords.words('german')
german_stopwords_wo_umlaut = []
for word in german_stopwords:
  german_stopwords_wo_umlaut.append(remove_umlauts(word))


pdf_url = 'https://dip21.bundestag.de/dip21/btp/15/15014.pdf'
filename = 'pdf_files/testfile'

#download_pdf(pdf_url, filename)
text = convert_pdf_to_text(filename)

cleaned_text = data_cleaning(text)

with_cleaning_chain = ''
for word in cleaned_text:
  with_cleaning_chain += word + ' '

n_gram_range = (1, 1)

# Extract candidate words/phrases
count = CountVectorizer(ngram_range=n_gram_range, stop_words=german_stopwords_wo_umlaut).fit([with_cleaning_chain])
candidates = count.get_feature_names()

model = SentenceTransformer('distilbert-base-nli-mean-tokens')
doc_embedding = model.encode([with_cleaning_chain])
candidate_embeddings = model.encode(candidates)  


top_n = 5
distances = cosine_similarity(doc_embedding, candidate_embeddings)
keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]

print(keywords)

import numpy as np

def mmr(doc_embedding, word_embeddings, words, top_n, diversity):

    # Extract similarity within words, and between words and the document
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)

    # Initialize candidates and already choose best keyword/keyphras
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(top_n - 1):
        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # Calculate MMR
        mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # Update keywords & candidates
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]

keywords2 = mmr(doc_embedding, candidate_embeddings, candidates, top_n=5, diversity=0.2)

print(keywords2)