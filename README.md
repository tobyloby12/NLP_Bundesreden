# Analysing the Bundestagreden using NLP techniques

## Abstract

## Introduction

Natural language processing is an important tool in helping machines understand and interpret complex sequences of text. Using these interpretations, predictions can be made on what the text is trying to convey. This has been demonstrated in the use of text summarization and keyword extraction. Using these techniques, large texts can be analysed and used to find more abstract meanings. This paper discusses the use of natural language processing on the Bundesreden and Coalition documents to predict potential projects that PD could be tasked with. 

## Method

### Inspecting the data 

The data used consists of a project databank of the PD past projects. Within the dataset there is a project title, description, and department of each project along with inception and completion dates if applicable. This data can be used to create examples of some of the projects that could be extracted from the Bundesreden. It can also be used to embed the descriptions and create embeddings which can then be compared with speeches to identify where the projects could have come from. Finally, the projects can be visualised by projecting the high dimensional embeddings into a lower three-dimensional vector.

The second dataset used consists of the Bundesreden. It has entries for the time the speech occurred and the date along with a web link to a pdf which contains the transcript of the speech. The data begins in 1949 meaning it is much larger than needed so changing the time frame to a more appropriate time was needed to get meaningful results. When looking at the pdf documents it is seen that they are extremely long and therefore some summarisation or keyword/keyphrase extraction is needed to condense the document down to get a better idea of what ideas are represented within it. 

Finally, a dataset was created by the author to list the departments at PD and have a description for each taken from the departments themselves. This dataset was used to be able to categorise which departments could be expecting potential projects predicted. It can also be used to see whether existing projects are similar to their respective department and see how relevant they are to each other.

### Text Extraction

Text extraction is essential as if the pdf documents were used, nothing could be used to put into an nlp algorithm. Therefore, extracting the text into a text file and then reading it is necessary to be able to easily access the data and pass it through various neural networks or algorithms to extract information. A function was written to download pdf files and save them from a URL. A second function was written to read the pdf and save the contents as a pdf file where each new line represents a sentence. These functions allow for all relevant speeches to be downloaded and made into a format that can be entered into a neural network for training or embedding.

### Text preparation and pre processing

After the text has been extracted from the pdf document it still has many issues with it which need to be resolved before it can be given to a network. This includes punctuation, special characters and such which don’t give the text any further information. These need to be removed so that the network can use the useful information without being confused or misled by these characters.

Stop words are words such as … which are common and don’t help with understanding the context or sentiment behind a piece of text. This means they need to be removed which will help with the text summarisation and keyword extraction as it removes common words and some keyword extraction algorithms use common words to decide what they want to use. The text also needs to be lemmatized meaning that all words are put into their base form. This allows similar words to be grouped together so that a network can see which words are most common and therefore most likely to be potential key words or phrases. In German accents are used so these need to be filtered out as they are considered special characters and can be easily substituted.

Finally, the text needs to be tokenised. This means splitting up the words so they can be seen as individual words within sentences instead of a continuous string of characters. This will allow a neural network to embed each word individually and compare them for applications such as predicting the next word or understanding the text and producing text from this.

### Transformers

A transformer is a deep learning technique which uses the mechanism of attention. This means that it places emphasis on specific words allowing for the model to consider context around key words allowing it to understand the text better and how words fit withing sentences. Transformers take in a sequence of data, such as a text document however doesn’t always process the data in that order. As it uses attention, certain words have more emphasis and are ‘remembered’ while less important words such as ‘der’ are not given much attention as they do not contribute to the understanding of the text. Transformers are used as a replacement for recurrent neural networks which will not be explained in this paper however for tasks that this paper concerns, using transformers will produce better results due to the superior text reasoning and understanding transformers possess.

### BERT

In 2018, Devlin et al. created and published a transformer machine learning technique called Bidirectional Encoder Representations from Transformers or BERT which was developed for Google. It has been shown to be very good at keyword extraction and other natural language understanding tasks. This model was therefore used for some of the applications which involved word embedding which is a technique of representing words in a high dimensional vector. The bidirectional aspect of the transformer means that it takes into account the whole sentence before determining which part requires the most attention and how words later in the sentence affect words coming before.
Keyword Extraction

There are two main types of machine learning, supervised and unsupervised learning. The former uses data which has labels meaning that it can be classified. This allows for a neural network or algorithm to be trained and it will then produce similar results to those given in training. The latter has no data labels meaning that it is just data. When looking at the data which was given, the speeches are not labelled data as potential projects cannot be assigned to certain speeches. Because the amount of data used is so large, it is not possible to create a labelled dataset without spending significant resources. Therefore, to begin with unsupervised machine learning techniques were used.

The first of these is keyword extraction. Keyword extraction is where the text is given into a neural network or algorithm and keywords or key phrases are extracted. Some techniques use methods such as finding the most common words while others also take the context into account to produce the most important parts of the document.

### Text Summarisation

Another method for extracting key information is text summarisation. The BERT algorithm has been found to be very good at understanding text and being able to analyse texts however struggles with creating new strings of text from it. Text summarisation requires looking at a piece of text and extracting the key information in a useful way. Autoregressive decoders such as the Seq2Seq model however are very good at decoding from vectors to words. Therefore, combining these two models will give us an encoder and decoder which can be used on our text. 

When using the text summarisation functions given by hugging face transformers library [ ], the speeches were passed through which had already been pre-processed by converting from pdf to text. It was found that the function could only take in tokens of 1024, therefore a function was written to split the text into its sentences and then into paragraphs which were no larger than 1024 words. After summarising the text, the document was embedded and compared to embeddings of the different previous projects that PD has completed using the cosine similarity.

### Visualisations

Using the PD website, the different departments in the company were identified and then description of each of these were found. The BERT network was used to embed these words into high dimensional vectors. These higher dimensional vectors are hard to understand therefore dimensional reduction would allow us to transform these higher degree vectors into lower dimensions such as two or three dimensional. This can then be plotted on a scatter plot to see where different departments lie within the lower dimensional vector space.

The dimensional reduction was achieved using Principal Component Analysis (PCA). This allowed for the higher 728 dimensional vectors to be projected onto 3 dimensions while losing as little accuracy and information as possible.

A second technique used was using T-SNE. This is a non-linear method of reducing the dimension of data and is better at creating clusters of similar data to get better separation. This is a large advantage over the PCA algorithm however it needs to be calibrated to the data and this can sometimes be challenging to achieve. When trying to calibrate the data for this project the visualisations were difficult to find and thus the PCA algorithm performed better. 


## Results

## Conclusion

## Explaination of different functions and how to use them based on each file:

### pdf_file_extraction.py

download_pdf(pdf_url, filename) - 

convert_text_to_pdf(filename) - 

### keyword_extraction.py

keyword_extraction(text, length) - this function takes in a text document and uses distilbert sentance transformers to extract key words. It also takes in a  length term which determines how many words long the keyword extraction string should be e.g. 1 for 1 keyword, 2 for a phrase of 2 keywords. It returns the keywords, the document embedding, candidate embedding and candidates which can be used for the mmr function.

mmr(doc_embedding, word_embeddings, words, top_n, diversity - this function uses keyword extraction and uses the cosine similarity to find how close together candidate keywords are to eachother so that the chosen keywords will be as diverse as possible. It takes in 4 terms which are the document embedding, word_embeddings which can be the candidate embeddings, words which are the candidate words, top_n which is the number of keyword phrases, diversity which is how dissimilar the words should be from eachother. It returns a set of keywords.

### summarisation_bert.py

remove_ulauts(word)

remove_currency(word)

lemmatizer(text)

data_cleaning(input_text)

summarisation(filename, input_text, total_length, MIN_length = 1, MAX_length = 300)

split_to_sentance(max_sentance_length, text)

### visualisations.py
