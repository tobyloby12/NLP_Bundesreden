import pandas as pd
from pdf_file_extraction import *
from transformers import pipeline
from summarisation_bert import *

# df_data = pd.read_excel('Datasets/PD-Projektdatenbank.xlsx')
# df_pd = df_data[['Titel', 'Projektbeschreibung', 'Leistungszeitraum']]
# df_speeches = pd.read_csv('Datasets/speeches.csv')

# df_speeches.date[2][:-6] == '1949'

# date_filter = []
# for line in range(len(df_speeches)):
#   date = int(df_speeches.date[line][:-6])
#   if date > 2000:
#     date_filter.append(True)
#   else:
#     date_filter.append(False)

# pd.DataFrame(date_filter)
# speeches_date_filtered = df_speeches[date_filter]


filename = 'pdf_files/testfile'


text = convert_pdf_to_text(filename)
text_split = split_to_sentance(300, text)
summarisation('coalition_summary.txt', text_split, len(text.split()), 1, 15)