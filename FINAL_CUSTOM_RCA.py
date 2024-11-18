

import numpy as np


import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import nltk
from nltk.stem import PorterStemmer
# from textblob import TextBlob
from tqdm import tqdm
import re

from bertopic import BERTopic


df=pd.read_csv('pmodata_300k.csv')

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

nltk.download('omw-1.4')



stopwords1=set(stopwords.words('english'))
stopwords2=set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"])

all_stopwords=set(stopwords1|stopwords2)

lemmatizer=WordNetLemmatizer()
ps = PorterStemmer()
def decontract(sentence):
    sentence=re.sub(r"n\'t"," not",sentence)
    sentence=re.sub(r"\'re"," are",sentence)
    sentence=re.sub(r"\'s"," is",sentence)
    sentence=re.sub(r"\'d"," would",sentence)
    sentence=re.sub(r"\'ll"," will",sentence)
    sentence=re.sub(r"\'ve"," have",sentence)
    sentence=re.sub(r"\'m"," am",sentence) 
    return sentence

def preprocess(desc):
    desc=re.sub(r"http\S+","",desc)
    desc=decontract(desc)
    desc=re.sub("\S*\d\S*","",desc).strip()
    desc=re.sub('[^A-Za-z]+',' ',desc)
    desc=' '.join(m.lower() for m in word_tokenize(desc) if m.lower() not in all_stopwords)
    desc=' '.join(lemmatizer.lemmatize(m,pos='a') for m in word_tokenize(desc) if m not in all_stopwords and len(m)>2)
    return desc





data['description'] = data['description'].apply(preprocess)




from sentence_transformers import SentenceTransformer, models

model_name = 'all-mpnet-base-v2'  # This model is more powerful than MiniLM
embedding_model = SentenceTransformer(model_name)
embeddings = embedding_model.encode(data['description'].tolist(), show_progress_bar=True, batch_size=32)

print("Embedding of the first document:", embeddings[0])
print("Shape of embeddings:", embeddings.shape)



def get_embeddings_by_date_and_initforward(starting_date, ending_date, initforward_str, data, embeddings):
    # Convert the 'regdate' column to datetime
    data['regdate'] = pd.to_datetime(data['regdate'])

    # Filter the data based on the conditions
    filtered_data = data[(data['regdate'] >= pd.to_datetime(starting_date)) &
                         (data['regdate'] <= pd.to_datetime(ending_date)) &
                         (data['initforward'] == initforward_str)]

    # Get indices of filtered data
    indices = filtered_data.index.tolist()

    # Ensure indices are within the range of stored_embeddings length
    valid_indices = [index for index in indices if index < len(stored_embeddings)]

    # Retrieve embeddings for the filtered data
    obtained_embeddings = np.array(stored_embeddings)[valid_indices]

    # Retrieve descriptions for the filtered data
    descriptions = filtered_data.loc[valid_indices, 'description'].tolist()
    filtered_data = filtered_data.loc[valid_indices]

    return obtained_embeddings, descriptions

def custom_rca():
    # Inputs from user
    starting_date = input("Enter the starting date (YYYY-MM-DD): ")
    ending_date = input("Enter the ending date (YYYY-MM-DD): ")
    initforward_str = input("Enter the initforward string: ")
    data = df[df['initforward'].isna() == False]  # Filter data where 'initforward' is not NaN
    # Get the embeddings and descriptions
    obtained_embeddings, descriptions = get_embeddings_by_date_and_initforward(starting_date, ending_date, initforward_str, data, embeddings)
    
    topic_model = BERTopic()
    docs = descriptions
    doc_id = list(range(1, 7959))  # List of registration Numbers.....

    
    # Fit the BERTopic model on the dummy documents and obtained embeddings
    
    topics, _ = topic_model.fit_transform(docs, obtained_embeddings)
    
    # Extract the topics and their frequencies
    topic_info = topic_model.get_topic_info()
    
    # Get the top 25 topics
    top_topics = topic_info.head(26)  # Get 26 as one topic might be -1 (outliers)
    
    topic_dict = {}
    doc_topic_dict = {}  # Initialize a new dictionary to store document IDs for each topic
    
    # Populate doc_topic_dict with document IDs for each topic
    for doc_index, topic_number in enumerate(topics):
        if topic_number != -1:  # excluding the outlier cluster
            if topic_number in doc_topic_dict:
                doc_topic_dict[topic_number].append(doc_id[doc_index])
            else:
                doc_topic_dict[topic_number] = [doc_id[doc_index]]
    
    # Iterate through the top topics and add document IDs to topic_dict
    for topic in top_topics['Topic']:
        if topic != -1:  # excluding the outlier cluster
            # Get top 10 words for the topic
            topic_words = topic_model.get_topic(topic)[:10]
            # Extract only the words (not the scores)
            words = [word for word, _ in topic_words]
            # Combine words and document IDs
            topic_info = {
                'words': words,
                'doc_ids': doc_topic_dict.get(topic, [])
            }
            topic_dict[topic] = topic_info

    return topic_dict

results = custom_rca()

print(results)