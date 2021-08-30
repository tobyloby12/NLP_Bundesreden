import pandas as pd
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import matplotlib.pyplot as plt
import plotly.express as px


df_pd = pd.read_excel('Datasets/PD-Projektdatenbank.xlsx')
df_pd = df_pd[['Titel', 'Projektbeschreibung', 'PD-Bereich(e)']]
#df_pd = df_pd.rename(columns={'PD-Bereich(e)': 'PD_Bereiche'})
df_pd = df_pd.dropna()

PD_Bereiche = df_pd['PD-Bereich(e)'].unique()

mapping = {}
for i in range(len(PD_Bereiche)):
  mapping[str(PD_Bereiche[i])] = i

df_pd = df_pd.replace(mapping)


nlp = spacy.load("de_core_news_sm")

nltk.download('stopwords')
stop_words = stopwords.words("german")

model = SentenceTransformer('msmarco-distilbert-multilingual-en-de-v2-tmp-trained-scratch')

df_pd = df_pd.dropna()
df_pd['Embeddings'] = ''

for i, row in df_pd.iterrows():
  #print(row['Projektbeschreibung'])
  #print(df_pd['Embeddings'][i])
  row_embedding = model.encode([row['Projektbeschreibung']])
  df_pd['Embeddings'][i] = row_embedding

print(df_pd.head())

import numpy as np

word_embeddings = []
for i, row in df_pd.iterrows():
  word_embeddings.append(row['Embeddings'][0])

from sklearn.decomposition import PCA

pca = PCA(n_components=3)
pca.fit(word_embeddings)

x_pca = pca.transform(word_embeddings)



plt.scatter(x_pca[:,0], x_pca[:,1], c = df_pd['PD-Bereich(e)'])
plt.legend()
print(mapping)


fig = px.scatter_3d(x=x_pca[:,0], y=x_pca[:,1], z=x_pca[:,2], color=df_pd['PD-Bereich(e)'], hover_name=df_pd['Titel'], title='Showing where projects are located in 3D space')
fig.show()