# Import required libraries

from transformers import AutoTokenizer, AutoModel
import torch
import json
from sklearn.metrics.pairwise import cosine_similarity
import sys

modelName = 'sentence-transformers/bert-base-nli-mean-tokens'
tokenizer = AutoTokenizer.from_pretrained(modelName)
model = AutoModel.from_pretrained(modelName)

# opening the file in read mode 
myFile =  open(sys.argv[2], "r") 

# reading the file 
data = myFile.read()
query = data.split("\n") 

studFile = open(sys.argv[1],'r')
studData = studFile.read()
answer = studData.split("A)")
answer.pop(0)
n = len(answer)

dictStud = {}
qnaStud = {}
for i in range(0,n):

  answer_cleaned = answer[i][:-2]
 
  qnaStud = {query[i]:answer[i]}
  dictStud = {**dictStud,**qnaStud}

gptData = json.load(open('model.json'))
gptSentences = []
for data in gptData:
    gptSentences.append(gptData[data])
counter = 0

dictResult = {}
for key in dictStud:
    sentences = []
    questList = key.split(' ')
    ansList =  dictStud[key].split(' ')
    
    # Remove repetion of question parts in answer that falsely increases the accuracy percentage
    for i in questList:
        if i in ansList:
            ansList.remove(i)
    finalAnswer = ' '.join(ansList)

    # Add all sentences for similarity comparison
    sentences.append(finalAnswer)
    sentences.extend(gptSentences)
    tokens = {'input_ids':[],'attention_mask':[]}

    # tokenize the sentences
    for sentence in sentences:
        newTokens = tokenizer.encode_plus(sentence, max_length=300,truncation=True,padding= 'max_length',return_tensors='pt')
        tokens['input_ids'].append(newTokens['input_ids'][0])
        tokens['attention_mask'].append(newTokens['attention_mask'][0])
    tokens['input_ids'] = torch.stack(tokens['input_ids'])
    tokens['attention_mask'] = torch.stack(tokens['attention_mask'])
    
    outputs = model(**tokens)
    embeddings = outputs.last_hidden_state
    attention = tokens['attention_mask']

    mask = attention.unsqueeze(-1).expand(embeddings.shape).float()
    maskEmbeddings = embeddings * mask
    summed = torch.sum(maskEmbeddings,1)
    counts = torch.clamp(mask.sum(1), min=1e-9)
    meanPooled = summed/counts
    meanPooled=meanPooled.detach().numpy()
    
    # Calculate cosine similarity
    similarity = cosine_similarity(
        [meanPooled[0]], meanPooled[1:]
    )
    result = {'Question {}'.format(counter+1):str(round(similarity[0][counter]*100, 2)) + '%'}
    dictResult = {**dictResult,**result}
    counter+=1


with open("result.json", "w") as outfile: 
    json.dump(dictResult, outfile)
    

