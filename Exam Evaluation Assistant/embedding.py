# Import required libraries

from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import sys
import json,os
from openai import AzureOpenAI
from langchain.chains.question_answering import load_qa_chain
from typing_extensions import Concatenate

# Get key and endpoint from enviornment variables
openAiKey = os.environ.get('API_KEY')
endPoint = os.environ.get('AI_endPoint')

# Define OpenAI client
client = AzureOpenAI(
  api_key = openAiKey,  
  api_version = "2023-05-15",
  azure_endpoint = endPoint
)

fileName = sys.argv[1]
# print(fileName)
pdfReader = PdfReader(fileName)


# read text from pdf

rawText = ''
for i, page in enumerate(pdfReader.pages):
    content = page.extract_text()
    if content:
        rawText += content



# We need to split the text using Character Text Split such that it sshould not increse token size
textSplitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 100,
    length_function = len,
)
texts = textSplitter.split_text(rawText
)

# Embed all text to vectors
embeddings = AzureOpenAIEmbeddings(  api_key = openAiKey,  
  api_version = "2023-05-15",
  azure_endpoint = endPoint)

documentSearch = FAISS.from_texts(texts, embeddings)
documentSearch.save_local('vectors')


