# Import required libraries

from openai import AzureOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
import json
import sys,os

# Get key and endpoint from enviornment variables
openAiKey = os.environ.get('API_KEY')
endPoint = os.environ.get('AI_ENDPOINT')

# Define OpenAI client
client = AzureOpenAI(
  api_key = openAiKey,  
  api_version = "2023-05-15",
  azure_endpoint = endPoint
)

# Retrieve embedding stored
embeddings = AzureOpenAIEmbeddings(  api_key = openAiKey,  
  api_version = "2023-05-15",
  azure_endpoint = endPoint)
document_search = FAISS.load_local('vectors', embeddings)



# opening the file in read mode 

myFile = open(sys.argv[1])
  
# Read the file and create suitable prompt
data = myFile.read()
query = data.split("\n") 
dictModel = {}
qna = {}
for question in query:

    docs = document_search.similarity_search(question)
    cont = docs[0].page_content
    
    header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
        
    prompt = header + "".join(cont) + "\n\n Q: " + question + "\n A:"

    response = client.chat.completions.create(model="gpt-35-turbo", messages=[{"role": "user", "content": prompt}], temperature=0.1)
    chat_response = response.choices[0].message.content
    qna = {question:chat_response}
    dictModel = {**dictModel,**qna}




with open("model.json", "w") as outfile: 
    json.dump(dictModel, outfile)
