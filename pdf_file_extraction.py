import os
import urllib.request
import pdfplumber


def download_pdf(pdf_url, filename):
  response = urllib.request.urlopen(pdf_url)
  file = open(filename + ".pdf", 'wb')
  file.write(response.read())
  file.close()

def convert_pdf_to_text(filename):
  with pdfplumber.open(filename + '.pdf') as pdf:
    without_cleaning = ''
    for i in range(len(pdf.pages)-1):
      page = pdf.pages[i]
      if page.extract_text() != None:
        without_cleaning = without_cleaning + page.extract_text()
    return without_cleaning