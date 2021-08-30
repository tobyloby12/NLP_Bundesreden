# file where summarisation code will go which can later be called

# where main code will go

import nltk
from nltk.corpus import stopwords
import nltk
from string import punctuation
from string import digits
import spacy
import re
from string import ascii_lowercase as alphabet
from transformers import pipeline
from tqdm import tqdm




# removing umlauts function
def remove_umlauts(word):
  tempWord = word
  tempWord = tempWord.replace('ä', 'ae')
  tempWord = tempWord.replace('ö', 'oe')
  tempWord = tempWord.replace('ü', 'ue')
  tempWord = tempWord.replace('Ä', 'Ae')
  tempWord = tempWord.replace('Ö', 'Oe')
  tempWord = tempWord.replace('Ü', 'Ue')
  tempWord = tempWord.replace('ß', 'ss')
  return tempWord

#remove currency function
def remove_currency(word):
  tempWord = word
  tempWord = tempWord.replace('$', '')
  tempWord = tempWord.replace('€', '')
  tempWord = tempWord.replace('¥', '')
  tempWord = tempWord.replace('₹', '')
  tempWord = tempWord.replace('£', '')
  return tempWord

#lemmatization function
def lemmatizer(text): 
    sent = []
    doc = model_de(text)
    for word in doc:
        sent.append(word.lemma_)
    return " ".join(sent)



#full function to clean text from input text
def data_cleaning(input_text):
  
  #removing punctuation
  remove_pun = str.maketrans('', '', punctuation)
  text_wo_pun = input_text.translate(remove_pun)

  #removing digits
  remove_digits = str.maketrans('', '', digits)
  text_wo_num = text_wo_pun.translate(remove_digits)

  #removing currency
  text_wo_currency = remove_currency(text_wo_num)

  #lemmatization
  text_lemmatized = lemmatizer(text_wo_currency.lower())

  #removing stop words
  text_wo_stop_words = [word for word in text_lemmatized.split() if text_lemmatized.lower() not in german_stopwords_wo_umlaut]

  return text_wo_stop_words




#summarisation hugging face transformers functions

def summarisation(filename, input_text, total_length, MIN_length = 1, MAX_length = 300):
  file = open(filename, 'w')
  file.write('')
  i = 0
  total = total_length
  for paragraph in tqdm(input_text):
    to_tokenize = paragraph
    #print(i/total * 100)

    # Initialize the HuggingFace summarization pipeline
    summarizer = pipeline("summarization")
    summarized = summarizer(to_tokenize, min_length=MIN_length, max_length=MAX_length)
    file = open(filename, 'a')
    file.write(summarized[0]['summary_text'] + '\n')
    i = i + len(paragraph.split())




# function to split sentances into paragraphs


def split_to_sentance(max_sentance_length, text):
  split_text = re.split(f'[{punctuation}\n]', text)
  split_text = [sentance.strip() for sentance in split_text]
  [split_text.remove(sentance) for sentance in split_text if sentance in ['','f{punctuation}']]
  split_text = [remove_umlauts(word) for word in split_text]
  
  
  
  paragraphs = ['' for i in range(len(text.split())//max_sentance_length + 1)]
  i = 0
  for sentance in split_text:
    sentance_length = len(sentance.split())

    if len(paragraphs[i].split()) < (max_sentance_length - sentance_length):
      paragraphs[i] += ' ' + sentance
    else:
      i += 1
  return paragraphs


#setting up stopwords
#nltk.download('stopwords')
model_de = spacy.load('de_core_news_sm')
german_stopwords = stopwords.words('german')
german_stopwords_wo_umlaut = []
for word in german_stopwords:
  german_stopwords_wo_umlaut.append(remove_umlauts(word))


#speech_content = df_speeches[['speechContent']]

