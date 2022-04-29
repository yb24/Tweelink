#!/usr/bin/env python
# coding: utf-8

# Import Statements

# In[ ]:


pip install --upgrade gensim


# In[ ]:


import pandas as pd
import numpy as np
import csv
import json
from itertools import islice
from collections import OrderedDict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import nltk
from glob import glob
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from tqdm import tqdm
import pickle
import math
import re
from sklearn.model_selection import train_test_split
import operator
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('corpus')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
get_ipython().system(' pip install git+https://github.com/LIAAD/yake')
import yake
stop_words = stopwords.words('english')


# In[ ]:


import logging
import gensim
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import SparseTermSimilarityMatrix, WordEmbeddingSimilarityIndex


# In[ ]:


# import gensim.downloader as api
# model = api.load('word2vec-google-news-300')


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


file1 = open("/content/drive/MyDrive/Tweelink_Dataset/twitter_base_preprocessed.pkl", "rb")
df = pickle.load(file1)
file1.close()


# In[ ]:


df.head()


# In[ ]:


u_base_hashtag = input("Enter base hashtag: ")
u_time = input("Enter time: ")
u_location = input("Enter Location: ")


# In[ ]:


import datetime
tweet_query = []
format = '%Y-%m-%d'
u_present_date = datetime.datetime.strptime(u_time, format)
u_prev_date = u_present_date - datetime.timedelta(days=1)
u_next_date = u_present_date + datetime.timedelta(days=1)
df_query = df.loc[df['hashtags'].str.contains(u_base_hashtag) & df['Date_Only'].isin([str(u_present_date.date()), str(u_prev_date.date()), str(u_next_date.date())])]


# In[ ]:


df_query.head()


# In[ ]:


df_query


# In[ ]:


def keyword_extractor(dataset):
  preprocessed_vocabulary = dict()

  #Converting to lowercase
  def to_lower_case(text):
    text = text.lower()
    return text

  def remove_at_word(text):
    data = text.split()
    data = [d for d in data if d[0]!='@']
    text = ' '.join(data)
    return text

  def remove_hashtag(text):
    data = text.split()
    data = [d if (d[0]!='#' or len(d) == 1) else d[1:] for d in data]
    data = [d for d in data if d[0]!='#']
    text = ' '.join(data)
    return text

  def remove_URL(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r'bit.ly\S+', '', text, flags=re.MULTILINE)
    return text

  #Removing stopwords
  def remove_stopwords(text):
    stopword = stopwords.words('english')
    new_list = [x for x in text.split() if x not in stopword]
    return ' '.join(new_list)

  #Removing punctuations
  def remove_punctuations(text):
    punctuations = '''!()-[|]`{};:'"\,<>./?@#$=+%^&*_~'''
    new_list = ['' if x in punctuations else x for x in text.split()]
    new_list_final = []
    for token in new_list:
      new_token=""
      for char in token:
        if(char not in punctuations):
          new_token+=char
      if(len(new_token)!=0):
        new_list_final.append(new_token)
    return ' '.join(new_list_final)

  #Tokenization
  def tokenization(text):
    return word_tokenize(text)

  def pre_process(text):
    text = to_lower_case(text)
    text = remove_at_word(text)
    text = remove_hashtag(text)
    text = remove_URL(text)
    text = remove_stopwords(text)
    text = remove_punctuations(text)
    text = tokenization(text)
    for token in text:
      if token in preprocessed_vocabulary.keys():
        preprocessed_vocabulary[token] += 1
      else:
        preprocessed_vocabulary[token] = 1
    return text
  
  preprocessed_data = [pre_process(text) for text in dataset]

  #print(preprocessed_vocabulary)

  AOF_coefficient = sum(preprocessed_vocabulary.values())/len(preprocessed_vocabulary)
  vocabulary = {token.strip():preprocessed_vocabulary[token] for token in preprocessed_vocabulary.keys() if preprocessed_vocabulary[token] > AOF_coefficient and len(token.strip())}

  #print(vocabulary)

  final_tokens_per_tweet = []
  for data in preprocessed_data:
    final_tokens_per_tweet.append([token for token in data if token in vocabulary.keys()])

  #print(preprocessed_data)
  #print(final_tokens_per_tweet)

  word2id = dict()
  id2word = dict()
  vocabulary_size = len(vocabulary)
  count = 0
  for token in vocabulary.keys():
    word2id[token] = count
    id2word[count] = token
    count += 1

  #print(word2id)
  #print(id2word)

  directed_graph_adjacency_matrix = np.zeros((vocabulary_size, vocabulary_size))
  edge_weight_matrix = np.zeros((vocabulary_size, vocabulary_size))
  first_frequency = dict()
  last_frequency = dict()
  term_frequency = vocabulary
  strength = dict()
  degree = dict()
  selective_centraility = dict()


  for tweet in final_tokens_per_tweet:
    if len(tweet) == 0:
      continue
    if tweet[0] in first_frequency.keys():
      first_frequency[tweet[0]] += 1
    else:
      first_frequency[tweet[0]] = 1

    if tweet[-1] in last_frequency.keys():
      last_frequency[tweet[-1]] += 1
    else:
      last_frequency[tweet[-1]] = 1
    


    for i in range(len(tweet)-1):
      if tweet[i] == tweet[i+1]:
        continue
      x = word2id[tweet[i]]
      y = word2id[tweet[i+1]]
      directed_graph_adjacency_matrix[x][y] += 1

  for tweet in final_tokens_per_tweet:
    for i in range(len(tweet)-1):

      if tweet[i] == tweet[i+1]:
        continue
      x = word2id[tweet[i]]
      y = word2id[tweet[i+1]]

    # Updating degree..
      if tweet[i] in degree.keys():
        degree[tweet[i]] += 1
      else:
        degree[tweet[i]] = 1
        
      if tweet[i+1] in degree.keys():
        degree[tweet[i+1]] += 1
      else:
        degree[tweet[i+1]] = 1

      edge_weight_matrix[x][y] = directed_graph_adjacency_matrix[x][y]/(vocabulary[tweet[i]] + vocabulary[tweet[i+1]] - directed_graph_adjacency_matrix[x][y])

      if tweet[i] in strength.keys():
        strength[tweet[i]] += edge_weight_matrix[x][y]
      else:
        strength[tweet[i]] = edge_weight_matrix[x][y]




  first_frequency = {token:(first_frequency[token]/vocabulary[token] if token in first_frequency else 0) for token in vocabulary.keys()}
  last_frequency = {token:(last_frequency[token]/vocabulary[token] if token in last_frequency else 0) for token in vocabulary.keys()}
  degree = {token:(degree[token] if token in degree else 0) for token in vocabulary.keys()}
  strength = {token:(strength[token] if token in strength else 0) for token in vocabulary.keys()}
  selective_centraility = {token:(strength[token]/degree[token] if degree[token]!=0 else 0) for token in vocabulary.keys()}

  #print(degree)
  #print(vocabulary)

  maxdegree = max(degree.items(), key=lambda x: x[1])[1]
  max_degree_nodes_with_freq = {key:term_frequency[key] for key in degree.keys() if degree[key] == maxdegree}
  maxfreq = max(max_degree_nodes_with_freq.items(), key=lambda x: x[1])[1]
  central_node_name = [key for key in max_degree_nodes_with_freq.keys() if max_degree_nodes_with_freq[key] == maxfreq][0]
  #print("central node: ", central_node_name)

  # bfs
  distance_from_central_node = dict()
  central_node_id = word2id[central_node_name]
  q = [(central_node_id, 0)]

  # Set source as visited
  distance_from_central_node[central_node_name] = 0

  while q:
      vis = q[0]
      # Print current node
      #print(id2word[vis[0]], vis[1])
      q.pop(0)
        
      # For every adjacent vertex to
      # the current vertex
      for i in range(len(directed_graph_adjacency_matrix[vis[0]])):
          if (directed_graph_adjacency_matrix[vis[0]][i] == 1 and (id2word[i] not in distance_from_central_node.keys())):
              # Push the adjacent node
              # in the queue
              q.append((i, vis[1]+1))
              distance_from_central_node[id2word[i]] = vis[1]+1

  #print(distance_from_central_node)
  inverse_distance_from_central_node = {token:(1/distance_from_central_node[token] if token in distance_from_central_node and token != central_node_name else 0) for token in vocabulary.keys()}
  inverse_distance_from_central_node[central_node_name] = 1.0
  #print(inverse_distance_from_central_node)

  neighbour_importance = dict()

  for i in range(len(directed_graph_adjacency_matrix)):
    neighbours = set()

    # traversing outgoing edges
    for j in range(len(directed_graph_adjacency_matrix)):
      if i == j:
        continue
      if directed_graph_adjacency_matrix[i][j] > 0:
        neighbours.add(j)
    for j in range(len(directed_graph_adjacency_matrix)):
      if i == j:
        continue
      if directed_graph_adjacency_matrix[j][i] > 0:
          neighbours.add(j)
    if len(neighbours) != 0:
      neighbour_importance[id2word[i]] = sum([strength[id2word[j]] for j in neighbours])/len(neighbours)
    else:
      neighbour_importance[id2word[i]] = 0
      
  #print(neighbour_importance)

  unnormalized_node_weight = {node: (first_frequency[node] + last_frequency[node] + term_frequency[node] + selective_centraility[node] + inverse_distance_from_central_node[node] + neighbour_importance[node]) for node in vocabulary.keys()}
  max_node_weight = max(unnormalized_node_weight.items(), key=lambda x: x[1])[1]
  min_node_weight = min(unnormalized_node_weight.items(), key=lambda x: x[1])[1]
  #print("max node weight: ", max_node_weight, "min node weight: ", min_node_weight)
  normalized_node_weight = {node: ((unnormalized_node_weight[node] - min_node_weight)/(max_node_weight - min_node_weight) if max_node_weight != min_node_weight else unnormalized_node_weight[node]) for node in unnormalized_node_weight.keys()}
  #print("Unnormalized score: ", unnormalized_node_weight)
  #print("Normalized score: ", normalized_node_weight)

  damping_factor = 0.85
  relevance_of_node = {node: np.random.uniform(0,1,1)[0] for node in vocabulary.keys()}
  threshold = 0.000000001


  #print(relevance_of_node)

  count = 0
  while True:
    count += 1
    current_relevance_of_node = dict()
    for node in vocabulary.keys():
      outer_sum = 0
      node_idx = word2id[node]
      for j in range(len(directed_graph_adjacency_matrix)):
        if j == node_idx:
          continue
        if directed_graph_adjacency_matrix[j][node_idx] > 0:
          den_sum = 0
          for k in range(len(directed_graph_adjacency_matrix)):
            if k == j:
              continue
            den_sum += directed_graph_adjacency_matrix[j][k]
          outer_sum += ((directed_graph_adjacency_matrix[j][node_idx]/den_sum) * relevance_of_node[id2word[j]])
      current_relevance_of_node[node] = (1-damping_factor)*normalized_node_weight[node] + damping_factor*normalized_node_weight[node]*outer_sum
    

    # checking convergence..
    sq_error = sum([(current_relevance_of_node[node] - relevance_of_node[node])**2 for node in vocabulary.keys()])
    relevance_of_node = current_relevance_of_node
    if sq_error < threshold:
      break

  #print(relevance_of_node)
  #print(count)

  degree_centrality  = {node: 0 for node in vocabulary.keys()}

  if len(directed_graph_adjacency_matrix) > 1:
    for i in range(len(directed_graph_adjacency_matrix)):
      count = 0
      for j in range(len(directed_graph_adjacency_matrix)):
        if i == j:
          continue
        if directed_graph_adjacency_matrix[j][i] > 0:
          count += 1
      degree_centrality[id2word[i]] = count / (len(directed_graph_adjacency_matrix)-1)

  #print(degree_centrality)

  final_keyword_rank = [{'node': node, 'NE_rank': relevance_of_node[node], 'Degree': degree_centrality[node]} for node in vocabulary.keys()]

  #print("-----------")
  final_keyword_rank = sorted(final_keyword_rank, key = lambda i: (i['NE_rank'], i['Degree']), reverse = True)

  final_keywords = [keyword['node'] for keyword in final_keyword_rank]

  return final_keywords


# In[ ]:


for tweet in df_query['Preprocessed_Data']:
  tweet_query.extend(tweet)


# In[ ]:


tweet_query


# In[ ]:


keyword_dataset = df_query['tweet'].tolist()
tweet_query_keyword_extractor = keyword_extractor(keyword_dataset)


# In[ ]:


print(keyword_dataset[0])
print(len(tweet_query))


# In[ ]:


tweet_query_keyword_extractor


# In[ ]:


tweet_keywords_yake = []
kw_extractor = yake.KeywordExtractor(top=20, stopwords=None)
keywords = kw_extractor.extract_keywords(' '.join(tweet_query))
#keywords = kw_extractor.extract_keywords(' '.join(df_query['tweet'].tolist()))
for kw, v in keywords:
  print("Keyphrase: ",kw, ": score", v)
  for key in kw.split():
    if(key.lower() not in tweet_keywords_yake):
      tweet_keywords_yake.append(key.lower())

print(tweet_keywords_yake)


# In[ ]:


docs_preprocessed = []


# In[ ]:


#Storing file name and data
total_documents = 0
path = '/content/drive/MyDrive/Tweelink_Dataset/Tweelink_Articles_Processed'
for filename in glob(os.path.join(path, '*')):
   with open(os.path.join(os.getcwd(), filename), 'r', encoding = 'utf-8',errors = 'ignore') as f:
     filename = os.path.basename(f.name)
     data = json.load(f)
     d_date = data["Date"]
     if(d_date=="" or d_date=="Date"):
       continue
     format = '%Y-%m-%d'
 
     d_present_date = datetime.datetime.strptime(d_date, format)
 
     if(str(d_present_date.date()) not in [str(u_present_date.date()), str(u_prev_date.date()), str(u_next_date.date())]):
       continue
   
     docs_preprocessed.append({'Name':filename, 'Data':data})
     total_documents+=1
print(total_documents)


# In[ ]:


print(docs_preprocessed[0])


# In[ ]:


def get_relevant_docs_list_for_base_hashtag(base_hashtag, base_date, docs_preprocessed):
  relevant_docs_list = []
  for doc in docs_preprocessed:
    if doc['Data']['Base Hashtag']==base_hashtag:
      current_date = datetime.datetime.strptime(base_date, format)
      prev_date = current_date - datetime.timedelta(days=1)
      next_date = current_date + datetime.timedelta(days=1)
      if(doc['Data']['Date'] in [str(prev_date.date()), str(current_date.date()), str(next_date.date())]):
        relevant_docs_list.append(doc['Name'])
  return relevant_docs_list


# In[ ]:


def precision_at_k(k, base_hashtag, base_date, prediction_list, docs_preprocessed):
  relevant_docs_list = get_relevant_docs_list_for_base_hashtag(base_hashtag, base_date, docs_preprocessed)
  num_of_relevant_results=0
  for itr in range(k):
    if (prediction_list[itr][0] in relevant_docs_list):
      num_of_relevant_results+=1
  return num_of_relevant_results/k


# In[ ]:


def mean_average_precision(max_k, base_hashtag, base_date, relevant_docs, docs_preprocessed):
  average_precision=0
  ctr=0
  for k_val in range(1,max_k+1):
    ctr+=1
    precision_at_k_val = precision_at_k(k_val, base_hashtag, base_date, relevant_docs, docs_preprocessed)
    #print('Hashtag: {}   Precision@{}: {}'.format(base_hashtag, k_val, precision_at_k_val))
    average_precision += precision_at_k_val
  return average_precision/ctr


# In[ ]:


def recall_at_k(k, base_hashtag, base_date, prediction_list, docs_preprocessed):
  relevant_docs_list = get_relevant_docs_list_for_base_hashtag(base_hashtag, base_date, docs_preprocessed)
  current_num_of_relevant_results=0
  for itr in range(k):
    if (prediction_list[itr][0] in relevant_docs_list):
      current_num_of_relevant_results+=1
  if(len(relevant_docs_list)==0):
    return 0
  return current_num_of_relevant_results/len(relevant_docs_list)


# In[ ]:


def mean_average_recall(max_k, base_hashtag, base_date, relevant_docs, docs_preprocessed):
  average_recall=0
  ctr=0
  for k_val in range(1,max_k+1):
    ctr+=1
    recall_at_k_val = recall_at_k(k_val, base_hashtag, base_date, relevant_docs, docs_preprocessed)
    #print('Hashtag: {}   Recall@{}: {}'.format(base_hashtag, k_val, recall_at_k_val))
    average_recall += recall_at_k_val
  return average_recall/ctr


# In[ ]:


# def find_relevant_documents_cosine_similarity_count_vectorizer(docs_preprocessed, processed_query):
#   cosine_similarities_cv = {}
#   for document in docs_preprocessed:
#     query_sent = ' '.join(map(str, processed_query))
#     doc_text_sent = ' '.join(map(str, document['Data']['Body_processed']))
#     data = [query_sent, doc_text_sent]
#     count_vectorizer = CountVectorizer(encoding='latin-1', decode_error='ignore', ngram_range=(1,2))
#     vector_matrix = count_vectorizer.fit_transform(data)
#     cosine_similarity_matrix = cosine_similarity(vector_matrix)
#     cosine_similarities_cv[document['Name']] = cosine_similarity_matrix[0][1]
#   relevant_docs = list( sorted(cosine_similarities_cv.items(), key=operator.itemgetter(1),reverse=True))[:20]
#   for i in range(len(relevant_docs)):
#     for j in range(len(docs_preprocessed)):
#       if(relevant_docs[i][0] == docs_preprocessed[j]['Name']):
#         relevant_docs[i] = (relevant_docs[i][0], relevant_docs[i][1], docs_preprocessed[j]['Data']['Date'] )

#   return relevant_docs


# In[ ]:


# def find_relevant_documents_cosine_similarity_tfidf_vectorizer(docs_preprocessed, processed_query):
#   cosine_similarities_tfidf = {}
#   for document in docs_preprocessed:
#     query_sent = ' '.join(map(str, processed_query))
#     doc_text_sent = ' '.join(map(str, document['Data']['Body_processed']))
#     data = [query_sent, doc_text_sent]
#     Tfidf_vect = TfidfVectorizer(encoding='latin-1', decode_error='ignore', ngram_range=(1,2))
#     vector_matrix = Tfidf_vect.fit_transform(data)
#     cosine_similarity_matrix = cosine_similarity(vector_matrix)
#     cosine_similarities_tfidf[document['Name']] = cosine_similarity_matrix[0][1]
#   relevant_docs = list( sorted(cosine_similarities_tfidf.items(), key=operator.itemgetter(1),reverse=True))[:20]
#   for i in range(len(relevant_docs)):
#     for j in range(len(docs_preprocessed)):
#       if(relevant_docs[i][0] == docs_preprocessed[j]['Name']):
#         relevant_docs[i] = (relevant_docs[i][0], relevant_docs[i][1], docs_preprocessed[j]['Data']['Date'] )

#   return relevant_docs


# In[ ]:


print(tweet_query)
print(tweet_query_keyword_extractor)
print(tweet_keywords_yake)


# #### Making own Corpus and applying Soft Cosine

# In[ ]:


from gensim.similarities import SoftCosineSimilarity
from gensim.models import Word2Vec
from gensim import utils

# #Storing file name and data

# all_docs = []
# path = '/content/drive/MyDrive/Tweelink_Dataset/Tweelink_Articles_Processed'
# for filename in glob(os.path.join(path, '*')):
#    with open(os.path.join(os.getcwd(), filename), 'r', encoding = 'utf-8',errors = 'ignore') as f:
#      filename = os.path.basename(f.name)
#      data = json.load(f)
#      d_date = data["Date"]
#      if(d_date=="" or d_date=="Date"):
#        continue
#      format = '%Y-%m-%d'
 
#      d_present_date = datetime.datetime.strptime(d_date, format)
 
#     #  if(str(d_present_date.date()) not in [str(u_present_date.date()), str(u_prev_date.date()), str(u_next_date.date())]):
#     #    continue
#      all_docs.append({'Name':filename, 'Data':data})

all_tweet_keywords = ' '.join(tweet_query)

whole_corpus_token_form = [article_data['Data']['Body_processed'] for article_data in docs_preprocessed]
whole_corpus_token_form.append(tweet_query_keyword_extractor)
print(len(whole_corpus_token_form))
# whole_corpus_token_form = [article_data['Data']['Body_processed'] for article_data in all_docs]
# whole_corpus_token_form.append(tweet_query_keyword_extractor)


def find_relevant_documents_soft_cosine_similarity(docs_preprocessed, processed_query):
  model = Word2Vec(sentences = whole_corpus_token_form, min_count=2) 
  termsim_index = WordEmbeddingSimilarityIndex(model.wv)
  sc_dictionary = Dictionary(whole_corpus_token_form)
  bow_corpus = [sc_dictionary.doc2bow(document) for document in whole_corpus_token_form]
  similarity_matrix = SparseTermSimilarityMatrix(termsim_index, sc_dictionary)
  docsim_index = SoftCosineSimilarity(bow_corpus, similarity_matrix, num_best=30)
  relevant_docs = docsim_index[sc_dictionary.doc2bow(processed_query)]
  iter = 0
  diff = 0
  for i in range(len(relevant_docs)):
    if relevant_docs[i][0] >= len(all_docs):
      continue
    relevant_docs[iter] = (all_docs[relevant_docs[i][0]]['Name'], relevant_docs[i][1], all_docs[relevant_docs[i][0]]['Data']['Date'])
    diff = i - iter
    iter += 1
  print(relevant_docs)
  return relevant_docs[:len(relevant_docs)-diff]


# Plain Model (Soft Cosine Similarity)

# In[ ]:


# Plain Model without YAKE / Keyword Extraction (Cosine Similarity Count Vectorizer)
relevant_docs_sc_plain = find_relevant_documents_soft_cosine_similarity(docs_preprocessed, tweet_query)

for rank, doc in enumerate(relevant_docs_sc_plain):
  print('Rank: {} Relevant Document: {}'.format(rank+1,doc))

print()

mean_average_precision_hashtag_sc_plain = mean_average_precision(20, u_base_hashtag, u_time, relevant_docs_sc_plain, docs_preprocessed)
print('Mean Average Precision Plain Model (Cosine Similarity Count Vectorizer) : {}'.format(mean_average_precision_hashtag_sc_plain))

mean_average_recall_hashtag_sc_plain = mean_average_recall(20, u_base_hashtag, u_time, relevant_docs_sc_plain, docs_preprocessed)
print('Mean Average Recall Plain Model (Cosine Similarity Count Vectorizer) : {}'.format(mean_average_recall_hashtag_sc_plain))


# In[ ]:


pip install annoy


# In[ ]:


count = 0
for li in whole_corpus_token_form:
  if 'hijab' in li:
    print(li)
    count += 1
print(count)


# In[ ]:


from gensim.test.utils import common_texts as corpus, datapath
from gensim.corpora import Dictionary
from gensim.models import Word2Vec
from gensim.similarities import SoftCosineSimilarity, SparseTermSimilarityMatrix, WordEmbeddingSimilarityIndex
from gensim.similarities.annoy import AnnoyIndexer

model = Word2Vec(whole_corpus_token_form, vector_size=50, min_count=5)  # train word-vectors
dictionary = Dictionary(whole_corpus_token_form)
tfidf = TfidfModel(dictionary=dictionary)
words = [word for word, count in dictionary.most_common(100)]
print(words)
word_vectors = model.wv.vectors_for_all(words, allow_inference=False)  # produce vectors for words in corpus
indexer = AnnoyIndexer(model.wv, num_trees=2)  # use Annoy for faster word similarity lookups
termsim_index = WordEmbeddingSimilarityIndex(model.wv, kwargs={'indexer': indexer})
similarity_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary, tfidf)  # compute word similarities
tfidf_corpus = tfidf[[dictionary.doc2bow(document) for document in whole_corpus_token_form]]
docsim_index = SoftCosineSimilarity(tfidf_corpus, similarity_matrix, num_best=10)  # index tfidf_corpus
relevant_docs = docsim_index[dictionary.doc2bow(tweet_query)]
print(relevant_docs)
iter = 0
diff = 0
for i in range(len(relevant_docs)):
  if relevant_docs[i][0] >= len(all_docs):
    continue
  relevant_docs[iter] = (all_docs[relevant_docs[i][0]]['Name'], relevant_docs[i][1], all_docs[relevant_docs[i][0]]['Data']['Date'])
  diff = i - iter
  iter += 1
print(relevant_docs)


# In[ ]:


for doc in docs_preprocessed:
  if doc['Name'] == 'MultiverseOfMadness_16.json':
    print(doc['Data'])


# Model with Keyword Extractor (Soft Cosine Similarity)

# In[ ]:


# Model with Keyword Extractor (Soft Cosine Similarity)
relevant_docs_sc_keyword_extractor = find_relevant_documents_soft_cosine_similarity(docs_preprocessed, tweet_query_keyword_extractor)

for rank, doc in enumerate(relevant_docs_sc_keyword_extractor):
  print('Rank: {} Relevant Document: {}'.format(rank+1,doc))

print()

mean_average_precision_hashtag_sc_keyword_extractor = mean_average_precision(20, u_base_hashtag, u_time, relevant_docs_sc_keyword_extractor, docs_preprocessed)
print('Mean Average Precision Plain Model (Cosine Similarity Count Vectorizer) : {}'.format(mean_average_precision_hashtag_sc_keyword_extractor))

mean_average_recall_hashtag_sc_keyword_extractor = mean_average_recall(20, u_base_hashtag, u_time, relevant_docs_sc_keyword_extractor, docs_preprocessed)
print('Mean Average Recall Plain Model (Cosine Similarity Count Vectorizer) : {}'.format(mean_average_recall_hashtag_sc_keyword_extractor))


# Model with YAKE (Soft Cosine Similarity)

# In[ ]:


# Model with YAKE (Soft Cosine Similarity)
relevant_docs_sc_yake = find_relevant_documents_soft_cosine_similarity(docs_preprocessed, tweet_keywords_yake)

for rank, doc in enumerate(relevant_docs_sc_yake):
  print('Rank: {} Relevant Document: {}'.format(rank+1,doc))

print()

mean_average_precision_hashtag_sc_yake = mean_average_precision(20, u_base_hashtag, u_time, relevant_docs_sc_yake, docs_preprocessed)
print('Mean Average Precision YAKE Model (Soft Cosine Similarity) : {}'.format(mean_average_precision_hashtag_sc_yake))

mean_average_recall_hashtag_sc_yake = mean_average_recall(20, u_base_hashtag, u_time, relevant_docs_sc_yake, docs_preprocessed)
print('Mean Average Recall YAKE Model (Soft Cosine Similarity) : {}'.format(mean_average_recall_hashtag_sc_yake))


# **Across all hashtags**

# In[ ]:


# global_list = [['narendramodi', '2022-02-14', 'india'],['UkraineRussiaCrisis', '2022-02-14', 'ukraine'],['IPL', '2022-02-14', 'india'],['TaylorSwift', '2022-02-14', 'USA'],['IndiaFightsCorona', '2022-02-14', 'india'],['narendramodi', '2022-02-15', 'india'],['UkraineRussiaCrisis', '2022-02-15', 'ukraine'],['IPL', '2022-02-15', 'india'],['TaylorSwift', '2022-02-15', 'USA'],['IndiaFightsCorona', '2022-02-15', 'india'],['narendramodi', '2022-02-16', 'india'],['UkraineRussiaCrisis', '2022-02-16', 'ukraine'],['IPL', '2022-02-16', 'india'],['TaylorSwift', '2022-02-16', 'USA'],['IndiaFightsCorona', '2022-02-16', 'india'],['narendramodi', '2022-02-17', 'india'],['UkraineRussiaCrisis', '2022-02-17', 'ukraine'],['IPL', '2022-02-17', 'india'],['TaylorSwift', '2022-02-17', 'USA'],['IndiaFightsCorona', '2022-02-17', 'india'],['narendramodi', '2022-02-18', 'india'],['UkraineRussiaCrisis', '2022-02-18', 'ukraine'],['IPL', '2022-02-18', 'india'],['TaylorSwift', '2022-02-18', 'USA'],['IndiaFightsCorona', '2022-02-18', 'india'],['narendramodi', '2022-02-19', 'india'],['UkraineRussiaCrisis', '2022-02-19', 'ukraine'],['IPL', '2022-02-19', 'india'],['hijab', '2022-02-19', 'india'],['vaccine', '2022-02-19', 'india'],['MillionAtIndiaPavilion', '2022-02-14', 'UAE'],['PunjabPanjeNaal', '2022-02-14', 'India'],['Euphoria', '2022-02-14', 'World'],['OscarsFanFavorite', '2022-02-14', 'World'],['ShameOnBirenSingh', '2022-02-14', 'india'],['BappiLahiri', '2022-02-16', 'india'],['BlandDoritos', '2022-02-16', 'USA'],['VERZUZ', '2022-02-16', 'USA'],['DragRaceUK', '2022-02-16', 'United Kingdom'],['BoycottWalgreens', '2022-02-18', 'USA'],['PunjabElections2022', '2022-02-20', 'india'],['WriddhimanSaha', '2022-02-20', 'india'],['stormfranklin', '2022-02-20', 'USA'],['QueenElizabeth', '2022-02-20', 'United Kingdom'],['ScottyFromWelding', '2022-02-20', 'Australia'],['CarabaoCupFinal', '2022-02-27', 'London'],['NZvSA', '2022-02-28', 'New Zealand'],['IPCC', '2022-02-28', 'Worldwide'],['SuperBowl', '2022-02-14', 'USA'],['MultiverseOfMadness', '2022-02-14', 'USA'],['Eminem', '2022-02-14', 'USA'],['IPLAuction', '2022-02-14', 'india'],['JohnsonOut21', '2022-02-14', 'United Kingdom'],['Cyberpunk2077', '2022-02-15', 'Worldwide'],['Wordle242', '2022-02-15', 'Worldwide'],['DeepSidhu', '2022-02-15', 'india'],['CanadaHasFallen', '2022-02-15', 'canada'],['IStandWithTrudeau', '2022-02-15', 'canada'],['CNNPHVPDebate', '2022-02-26', 'philippines'],['qldfloods', '2022-02-26', 'australia'],['Eurovision', '2022-02-26', 'worldwide'],['IndiansInUkraine', '2022-02-26', 'india'],['PritiPatel', '2022-02-26', 'united kingdom'],['TaylorCatterall', '2022-02-27', 'united kingdom'],['PSLFinal', '2022-02-27', 'pakistan'],['AustraliaDecides', '2022-02-27', 'australia'],['WorldNGODay', '2022-02-27', 'worldwide'],['TheBatman', '2022-02-28', 'USA'],['NationalScienceDay', '2022-02-28', 'india'],['msdtrong', '2022-02-14', 'india'],['Boycott_ChennaiSuperKings', '2022-02-14', 'india'],['GlanceJio', '2022-02-14', 'india'],['ArabicKuthu', '2022-02-14', 'india'],['Djokovic', '2022-02-15', 'australia'],['Real Madrid', '2022-02-15', 'santiago'],['bighit', '2022-02-15', 'korea'],['Maxwell', '2022-02-15', 'australia'],['mafsau', '2022-02-16', 'australia'],['channi', '2022-02-16', 'punjab'],['ayalaan', '2022-02-16', 'india'],['jkbose', '2022-02-16', 'india'],['HappyBirthdayPrinceSK', '2022-02-16', 'india'],['RandomActsOfKindnessDay', '2022-02-17', 'worldwide'],['happybirthdayjhope', '2022-02-17', 'korea'],['mohsinbaig', '2022-02-17', 'pakistan'],['aewdynamite', '2022-02-17', 'worldwide'],['aaraattu', '2022-02-17', 'india'],['ShivajiJayanti', '2022-02-18', 'india'],['PlotToKillModi', '2022-02-18', 'india'],['NationalDrinkWineDay', '2022-02-18', 'usa'],['HorizonForbiddenWest', '2022-02-18', 'worldwide'],['BoycottWalgreens', '2022-02-18', 'usa'],['CallTheMidwife', '2022-02-20', 'worldwide'],['OperationDudula', '2022-02-20', 'south africa'],['truthsocial', '2022-02-21', 'usa'],['nbaallstar', '2022-02-21', 'usa'],['shivamogga', '2022-02-21', 'india'],['HalftimeShow', '2022-02-14', 'usa'],['OttawaStrong', '2022-02-14', 'canada'],['DrDre', '2022-02-14', 'usa'],['BattleOfBillingsBridge', '2022-02-14', 'usa'],['FullyFaltooNFTdrop', '2022-02-14', 'worldwide'],['AK61', '2022-02-15', 'india'],['sandhyamukherjee', '2022-02-15', 'india'],['MUNBHA', '2022-02-15', 'worldwide'],['nursesstrike', '2022-02-15', 'australia'],['Realme9ProPlus', '2022-02-16', 'worldwide'],['KarnatakaHijabControversy', '2022-02-16', 'india'],['BJPwinningUP', '2022-02-16', 'india'],['Punjab_With_Modi', '2022-02-16', 'india'],['PushpaTheRule', '2022-02-16', 'india'],['RehmanMalik', '2022-02-22', 'india'],['harisrauf', '2022-02-22', 'pakistan'],['Rosettenville', '2022-02-22', 'south africa'],['NFU22', '2022-02-22', 'worldwide'],['justiceforharsha', '2022-02-22', 'india'],['wordle251', '2022-02-24', 'worldwide'],['ARSWOL', '2022-02-24', 'worldwide'],['stopwar', '2022-02-24', 'worldwide'],['PrayForPeace', '2022-02-24', 'worldwide'],['StopPutinNOW', '2022-02-24', 'worldwide'],['TeamGirlsCup', '2022-02-25', 'worldwide'],['Canucks', '2022-02-25', 'worldwide'],['PinkShirtDay', '2022-02-25', 'canada'],['superrugbypacific', '2022-02-25', 'australia']]


# In[ ]:


# global_average_mean_average_precision_cs_cv = []
# global_mean_average_recall_cs_cv = []

# global_average_mean_average_precision_cs_tfidf = []
# global_mean_average_recall_cs_tfidf = []

# for iter in tqdm(range(len(global_list))):
#   u_base_hashtag = global_list[iter][0]
#   u_time = global_list[iter][1]
#   u_location = global_list[iter][2]
#   tweet_query = []
#   format = '%Y-%m-%d'
#   u_present_date = datetime.datetime.strptime(u_time, format)
#   u_prev_date = u_present_date - datetime.timedelta(days=1)
#   u_next_date = u_present_date + datetime.timedelta(days=1)
#   df_query = df.loc[df['hashtags'].str.contains(u_base_hashtag) & df['Date_Only'].isin([str(u_present_date.date()), str(u_prev_date.date()), str(u_next_date.date())])]

#   for tweet in df_query['Preprocessed_Data']:
#     tweet_query.extend(tweet)
  
#   # tweet_keywords = []
#   # kw_extractor = yake.KeywordExtractor(top=20, stopwords=None)
#   # keywords = kw_extractor.extract_keywords(' '.join(tweet_query))
#   # for kw, v in keywords:
#   #   #print("Keyphrase: ",kw, ": score", v)
#   #   for key in kw.split():
#   #     if(key not in tweet_keywords):
#   #       tweet_keywords.append(key)
  
#   docs_preprocessed = []

#   total_documents = 0
#   path = '/content/drive/MyDrive/Tweelink_Dataset/Tweelink_Articles_Processed'
#   for filename in glob(os.path.join(path, '*')):
#     with open(os.path.join(os.getcwd(), filename), 'r', encoding = 'utf-8',errors = 'ignore') as f:
#       filename = os.path.basename(f.name)
#       data = json.load(f)
#       d_date = data["Date"]
#       if(d_date=="" or d_date=="Date"):
#         continue
#       format = '%Y-%m-%d'
  
#       d_present_date = datetime.datetime.strptime(d_date, format)
  
#       if(str(d_present_date.date()) not in [str(u_present_date.date()), str(u_prev_date.date()), str(u_next_date.date())]):
#         continue
    
#       docs_preprocessed.append({'Name':filename, 'Data':data})
#       total_documents+=1
  
#   relevant_docs_cs_cv = find_relevant_documents_cosine_similarity_count_vectorizer(docs_preprocessed, tweet_query)
#   mean_average_precision_hashtag_cs_cv = mean_average_precision(20, u_base_hashtag, u_time, relevant_docs_cs_cv, docs_preprocessed)
#   global_average_mean_average_precision_cs_cv.append(mean_average_precision_hashtag_cs_cv)
#   mean_average_recall_hashtag_cs_cv = mean_average_recall(20, u_base_hashtag, u_time, relevant_docs_cs_cv, docs_preprocessed)
#   global_mean_average_recall_cs_cv.append(mean_average_recall_hashtag_cs_cv)

#   relevant_docs_cs_tfidf = find_relevant_documents_cosine_similarity_tfidf_vectorizer(docs_preprocessed, tweet_query)
#   mean_average_precision_hashtag_cs_tfidf = mean_average_precision(20, u_base_hashtag, u_time, relevant_docs_cs_tfidf, docs_preprocessed)
#   global_average_mean_average_precision_cs_tfidf.append(mean_average_precision_hashtag_cs_tfidf)
#   mean_average_recall_hashtag_cs_tfidf = mean_average_recall(20, u_base_hashtag, u_time, relevant_docs_cs_tfidf, docs_preprocessed)
#   global_mean_average_recall_cs_tfidf.append(mean_average_recall_hashtag_cs_tfidf)


# In[ ]:


# # cs cv
# overall_average_mean_average_precision_cs_cv = sum(global_average_mean_average_precision_cs_cv)/len(global_average_mean_average_precision_cs_cv)
# print(overall_average_mean_average_precision_cs_cv)

# overall_mean_average_recall_cs_cv = sum(global_mean_average_recall_cs_cv)/len(global_mean_average_recall_cs_cv)
# print(overall_mean_average_recall_cs_cv)


# In[ ]:


# # cs tfidf
# overall_average_mean_average_precision_cs_tfidf = sum(global_average_mean_average_precision_cs_tfidf)/len(global_average_mean_average_precision_cs_tfidf)
# print(overall_average_mean_average_precision_cs_tfidf)

# overall_mean_average_recall_cs_tfidf = sum(global_mean_average_recall_cs_tfidf)/len(global_mean_average_recall_cs_tfidf)
# print(overall_mean_average_recall_cs_tfidf)

