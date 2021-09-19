## Explaination of different functions and how to use them based on each file:

### pdf_file_extraction.py

download_pdf(pdf_url, filename) - A function that takes in the url to a pdf and the filename that the file should be saved to and downloads the pdf and saves this to the file.

convert_text_to_pdf(filename) - Takes a pdf file of name filename and converts it into a string with new lines being represented as new lines.

### keyword_extraction.py

keyword_extraction(text, length) - this function takes in a text document and uses distilbert sentance transformers to extract key words. It also takes in a  length term which determines how many words long the keyword extraction string should be e.g. 1 for 1 keyword, 2 for a phrase of 2 keywords. It returns the keywords, the document embedding, candidate embedding and candidates which can be used for the mmr function.

mmr(doc_embedding, word_embeddings, words, top_n, diversity - this function uses keyword extraction and uses the cosine similarity to find how close together candidate keywords are to eachother so that the chosen keywords will be as diverse as possible. It takes in 4 terms which are the document embedding, word_embeddings which can be the candidate embeddings, words which are the candidate words, top_n which is the number of keyword phrases, diversity which is how dissimilar the words should be from eachother. It returns a set of keywords.

### summarisation_bert.py

remove_ulauts(word) - function that replaces umlauts with different characters. This is needed to clean the text and let the algorithm understand the context.

remove_currency(word) - function to remove currency symbols and replace with nothing to allow for better text understanding.

lemmatizer(text) - lemmatises the text and makes words into their base form e.g. ran, runs -> run.

data_cleaning(input_text) - combines all previous functions in summarisation_bert.py and creates a seemless pipeline

summarisation(filename, input_text, total_length, MIN_length = 1, MAX_length = 300) - uses hugging face transformers library to summarise the text to a set length given byt the MIN_length and MAX_length texts.

split_to_sentance(max_sentance_length, text) - the summarisation function requires tokens of only 350 words therefore a function was written to split the text up into the correct sizes. It does this by splitting the text based on punctuation to form sentances and then puts sentances together so they do not pass the maximum number of tokens given by max_sentence_length.
