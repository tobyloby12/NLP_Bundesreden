# NLP_Bundesreden
Seeing how using NLP, potential projects for PD could be determined from Bundesreden

## Explaination of different functions and how to use them based on each file:

### pdf_file_extraction.py

download_pdf(pdf_url, filename) - 
convert_text_to_pdf(filename) - 

### keyword_extraction.py

keyword_extraction(text, length) - this function takes in a text document and uses distilbert sentance transformers to extract key words. It also takes in a  length term which determines how many words long the keyword extraction string should be e.g. 1 for 1 keyword, 2 for a phrase of 2 keywords. It returns the keywords, the document embedding, candidate embedding and candidates which can be used for the mmr function.

mmr(doc_embedding, word_embeddings, words, top_n, diversity - this function uses keyword extraction and uses the cosine similarity to find how close together candidate keywords are to eachother so that the chosen keywords will be as diverse as possible. It takes in 4 terms which are the document embedding, word_embeddings which can be the candidate embeddings, words which are the candidate words, top_n which is the number of keyword phrases, diversity which is how dissimilar the words should be from eachother. It returns a set of keywords.

### 
