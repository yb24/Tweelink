# -*- coding: utf-8 -*-
"""final_results_okapi_bm25.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1l4jSZhpUMV14Ic4mMu4rtb072wBYYMGN

# Import Statements
"""

#Importing essential libraries
import pandas as pd
import numpy as np
import csv
import json
from itertools import islice
from collections import OrderedDict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
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
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('corpus')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
# ! pip install git+https://github.com/LIAAD/yake
import yake
# ! pip install multi_rake
from multi_rake import Rake
# ! pip install summa
from summa import keywords as summa_keywords
# ! pip install keybert
from keybert import KeyBERT

from nltk.stem import WordNetLemmatizer
# nltk.download('wordnet')

# from google.colab import drive
# drive.mount('/content/drive')

# file1 = open("/content/drive/MyDrive/Tweelink_Dataset/twitter_base_preprocessed.pkl", "rb")
# df = pickle.load(file1)
# file1.close()

"""# Input"""

format = '%Y-%m-%d'

# u_base_hashtag = input("Enter base hashtag: ")
# u_time = input("Enter time: ")
# u_location = input("Enter Location: ")

"""# Processing"""

# import datetime
# tweet_query = []
# format = '%Y-%m-%d'
# u_present_date = datetime.datetime.strptime(u_time, format)
# u_prev_date = u_present_date - datetime.timedelta(days=1)
# u_next_date = u_present_date + datetime.timedelta(days=1)
# df_query = df.loc[df['hashtags'].str.contains(u_base_hashtag) & df['Date_Only'].isin([str(u_present_date.date()), str(u_prev_date.date()), str(u_next_date.date())])]
# print(df_query.shape[0])
# if df_query.shape[0]<50:
#   df_query = df.loc[df['Date_Only'].isin([str(u_present_date.date()), str(u_prev_date.date()), str(u_next_date.date())])]
#   df_query = df.iloc[:min(df_query.shape[0],1000),:]

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
    if len(tweet)<1:
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

# for tweet in df_query['Preprocessed_Data']:
#   tweet_query.extend(tweet)

# keyword_dataset = df_query['tweet'].tolist()
# tweet_query_keyword_extractor = keyword_extractor(keyword_dataset)

# tweet_keywords_yake = []
# kw_extractor = yake.KeywordExtractor(top=20, stopwords=None)
# keywords = kw_extractor.extract_keywords(' '.join(tweet_query))
# #keywords = kw_extractor.extract_keywords(' '.join(df_query['tweet'].tolist()))
# for kw, v in keywords:
#   print("Keyphrase: ",kw, ": score", v)
#   for key in kw.split():
#     if(key.lower() not in tweet_keywords_yake):
#       tweet_keywords_yake.append(key.lower())

# tweet_keywords_rake = []
# rake = Rake()
# rake_keywords = rake.apply(' '.join(tweet_query).encode('ascii', 'ignore').decode())
# for kw,score in rake_keywords[:20]:
#   for key in kw.split():
#     if(key.lower() not in tweet_keywords_yake):
#       tweet_keywords_rake.append(key.lower())

# tweet_keywords_text_rank = []
# TR_keywords = summa_keywords.keywords(' '.join(tweet_query), scores=True)
# for kw,score in TR_keywords[:20]:
#   for key in kw.split():
#     if(key.lower() not in tweet_keywords_text_rank):
#       tweet_keywords_text_rank.append(key.lower())
# print(tweet_keywords_text_rank)

# uncomment later
# keybert_model = KeyBERT(model='all-mpnet-base-v2')
# keybert_keywords = keybert_model.extract_keywords(' '.join(tweet_query), keyphrase_ngram_range=(1,1), stop_words='english', highlight=False, top_n=20)
# tweet_keywords_keybert = list(dict(keybert_keywords).keys())
# print(tweet_keywords_keybert)

"""# Helpful Functions"""

# docs_preprocessed = []

# #Storing file name and data
# total_documents = 0
# path = '/content/drive/MyDrive/Tweelink_Dataset/Tweelink_Articles_Processed'
# for filename in glob(os.path.join(path, '*')):
#    with open(os.path.join(os.getcwd(), filename), 'r', encoding = 'utf-8',errors = 'ignore') as f:
#      filename = os.path.basename(f.name)
#      data = json.load(f)
#      d_date = data["Date"]
#      if(d_date=="" or d_date=="Date"):
#        continue
#      format = '%Y-%m-%d'
 
#      try:
#        d_present_date = datetime.datetime.strptime(d_date, format)
#      except:
#        continue
 
#      if(str(d_present_date.date()) not in [str(u_present_date.date()), str(u_prev_date.date()), str(u_next_date.date())]):
#        continue
   
#      docs_preprocessed.append({'Name':filename, 'Data':data})
#      total_documents+=1
# print(total_documents)

import datetime
def get_relevant_docs_list_for_base_hashtag(base_hashtag, base_date, docs_preprocessed, u_location):
  relevant_docs_list = []
  for doc in docs_preprocessed:
    if doc['Data']['Base Hashtag']==base_hashtag:
      current_date = datetime.datetime.strptime(base_date, format)
      prev_date = current_date - datetime.timedelta(days=1)
      next_date = current_date + datetime.timedelta(days=1)
      #if(doc['Data']['Date'] in [str(prev_date.date()), str(current_date.date()), str(next_date.date())]):
      #  relevant_docs_list.append((doc['Name'], doc['Data']['Location'].lower()))
      if(doc['Data']['Date']==str(str(current_date.date()))):
        relevant_docs_list.append((doc['Name'], doc['Data']['Location'].lower(), 1.0))
      elif(doc['Data']['Date'] in [str(prev_date.date()), str(next_date.date())]):
        relevant_docs_list.append((doc['Name'], doc['Data']['Location'].lower(), 0.5))
  
  # prioritize location
  location_relevant = []
  location_irrelevant = []
  for x in relevant_docs_list:
    if u_location.lower() in x[1]:
      location_relevant.append(x)
    else:
      location_irrelevant.append(x)
  relevant_docs_list = location_relevant + location_irrelevant
  return relevant_docs_list

def nDCG(base_hashtag, base_date, prediction_list, docs_preprocessed):
  ground_truth = get_relevant_docs_list_for_base_hashtag(base_hashtag, base_date, docs_preprocessed)
  ground_truth_scores = {}
  ground_truth_scores_list = []
  prediction_list_scores = []
  for gt in ground_truth:
    ground_truth_scores[gt[0]] = gt[2]
    ground_truth_scores_list.append(gt[2])
  for x in prediction_list:
    if x[0] in ground_truth_scores.keys():
      prediction_list_scores.append(ground_truth_scores[x[0]])
    else:
      prediction_list_scores.append(0.0)
  
  DCG = prediction_list_scores[0] + sum([prediction_list_scores[i]/np.log2(i+1) for i in range(1,len(prediction_list_scores))])
  ideal_DCG = ground_truth_scores_list[0] + sum([ground_truth_scores_list[i]/np.log2(i+1) for i in range(1,len(ground_truth_scores_list))])
  if ideal_DCG==0:
    return DCG
  return DCG/ideal_DCG

def precision_at_k(k, base_hashtag, base_date, prediction_list, docs_preprocessed):
  relevant_docs_list = get_relevant_docs_list_for_base_hashtag(base_hashtag, base_date, docs_preprocessed)
  num_of_relevant_results=0
  list_of_rel_doc_names = []
  for x in relevant_docs_list:
    list_of_rel_doc_names.append(x[0])
  for itr in range(k):
    if (prediction_list[itr][0] in list_of_rel_doc_names):
      num_of_relevant_results+=1
  return num_of_relevant_results/k

def mean_average_precision(max_k, base_hashtag, base_date, relevant_docs, docs_preprocessed):
  average_precision=0
  ctr=0
  relevant_docs_list = get_relevant_docs_list_for_base_hashtag(base_hashtag, base_date, docs_preprocessed)
  print(len(relevant_docs_list))
  for k_val in range(1,len(relevant_docs_list)+1):
    ctr+=1
    if k_val>len(relevant_docs_list):
      break
    precision_at_k_val = precision_at_k(k_val, base_hashtag, base_date, relevant_docs, docs_preprocessed)
    #print('Hashtag: {}   Precision@{}: {}'.format(base_hashtag, k_val, precision_at_k_val))
    average_precision += precision_at_k_val
  return average_precision/ctr

def recall_at_k(k, base_hashtag, base_date, prediction_list, docs_preprocessed):
  relevant_docs_list = get_relevant_docs_list_for_base_hashtag(base_hashtag, base_date, docs_preprocessed)
  current_num_of_relevant_results=0
  list_of_rel_doc_names = []
  for x in relevant_docs_list:
    list_of_rel_doc_names.append(x[0])
  for itr in range(k):
    if (prediction_list[itr][0] in list_of_rel_doc_names):
      current_num_of_relevant_results+=1
  if(len(relevant_docs_list)==0):
    return 0
  return current_num_of_relevant_results/len(relevant_docs_list)

def mean_average_recall(max_k, base_hashtag, base_date, relevant_docs, docs_preprocessed):
  average_recall=0
  ctr=0
  relevant_docs_list = get_relevant_docs_list_for_base_hashtag(base_hashtag, base_date, docs_preprocessed)
  print(len(relevant_docs_list))
  for k_val in range(1,len(relevant_docs_list)+1):
    ctr+=1
    if k_val>len(relevant_docs_list):
      break
    recall_at_k_val = recall_at_k(k_val, base_hashtag, base_date, relevant_docs, docs_preprocessed)
    #print('Hashtag: {}   Recall@{}: {}'.format(base_hashtag, k_val, recall_at_k_val))
    average_recall += recall_at_k_val
  return average_recall/ctr

# ! pip install rank_bm25
from rank_bm25 import BM25Okapi


def find_relevant_documents_okapibm25(docs_preprocessed, processed_query, u_base_hashtag, u_time, u_location):
  max_list_size = len(get_relevant_docs_list_for_base_hashtag(u_base_hashtag, u_time, docs_preprocessed, u_location))
  tokenized_corpus = [doc['Data']['Body_processed'] for doc in docs_preprocessed]
  okapibm25_model = BM25Okapi(tokenized_corpus)
  doc_scores_okapibm25 = okapibm25_model.get_scores(processed_query)
  okapibm25_ids = doc_scores_okapibm25.argsort()[::-1][:max_list_size]
  relevant_docs_okapibm25 = []
  for idx in okapibm25_ids:
    relevant_docs_okapibm25.append((docs_preprocessed[idx]['Name'], doc_scores_okapibm25[idx], docs_preprocessed[idx]['Data']['Date'], docs_preprocessed[idx]['Data']['Location'].lower(), docs_preprocessed[idx]['Data']['Link']))

  # prioritize location
  location_relevant = []
  location_irrelevant = []
  for x in relevant_docs_okapibm25:
    if u_location.lower() in x[3]:
      location_relevant.append(x)
    else:
      location_irrelevant.append(x) 
  relevant_docs_okapibm25 = location_relevant + location_irrelevant
  return relevant_docs_okapibm25

# """# Plain Model (Okapi BM25)"""

# relevant_docs_okapibm25_plain = find_relevant_documents_okapibm25(docs_preprocessed, tweet_query)

# for rank, doc in enumerate(relevant_docs_okapibm25_plain):
#   print('Rank: {} Relevant Document: {}'.format(rank+1,doc))

# print()

# mean_average_precision_hashtag_okapibm25_plain = mean_average_precision(20, u_base_hashtag, u_time, relevant_docs_okapibm25_plain, docs_preprocessed)
# print('Mean Average Precision Plain Model (Okapi BM25) : {}'.format(mean_average_precision_hashtag_okapibm25_plain))

# mean_average_recall_hashtag_okapibm25_plain = mean_average_recall(20, u_base_hashtag, u_time, relevant_docs_okapibm25_plain, docs_preprocessed)
# print('Mean Average Recall Plain Model (Okapi BM25) : {}'.format(mean_average_recall_hashtag_okapibm25_plain))

# nDCG_hashtag_okapibm25_plain = nDCG(u_base_hashtag, u_time, relevant_docs_okapibm25_plain, docs_preprocessed)
# print('nDCG Plain Model (Okapi BM25) : {}'.format(nDCG_hashtag_okapibm25_plain))

# """# Model with Keyword Extractor (Okapi BM25)"""

# relevant_docs_okapibm25_keyword_extractor = find_relevant_documents_okapibm25(docs_preprocessed, tweet_query_keyword_extractor[:20])

# for rank, doc in enumerate(relevant_docs_okapibm25_keyword_extractor):
#   print('Rank: {} Relevant Document: {}'.format(rank+1,doc))

# print()

# mean_average_precision_hashtag_okapibm25_keyword_extractor = mean_average_precision(20, u_base_hashtag, u_time, relevant_docs_okapibm25_keyword_extractor, docs_preprocessed)
# print('Mean Average Precision Keyword Extractor Model (Okapi BM25) : {}'.format(mean_average_precision_hashtag_okapibm25_keyword_extractor))

# mean_average_recall_hashtag_okapibm25_keyword_extractor = mean_average_recall(20, u_base_hashtag, u_time, relevant_docs_okapibm25_keyword_extractor, docs_preprocessed)
# print('Mean Average Recall Keyword Extractor Model (Okapi BM25) : {}'.format(mean_average_recall_hashtag_okapibm25_keyword_extractor))

# nDCG_hashtag_okapibm25_keyword_extractor = nDCG(u_base_hashtag, u_time, relevant_docs_okapibm25_keyword_extractor, docs_preprocessed)
# print('nDCG Keyword Extractor Model (Okapi BM25) : {}'.format(nDCG_hashtag_okapibm25_keyword_extractor))

# """# Model with YAKE (Okapi BM25)"""

# relevant_docs_okapibm25_yake = find_relevant_documents_okapibm25(docs_preprocessed, tweet_keywords_yake)

# for rank, doc in enumerate(relevant_docs_okapibm25_yake):
#   print('Rank: {} Relevant Document: {}'.format(rank+1,doc))

# print()

# mean_average_precision_hashtag_okapibm25_yake = mean_average_precision(20, u_base_hashtag, u_time, relevant_docs_okapibm25_yake, docs_preprocessed)
# print('Mean Average Precision YAKE Model (Okapi BM25) : {}'.format(mean_average_precision_hashtag_okapibm25_yake))

# mean_average_recall_hashtag_okapibm25_yake = mean_average_recall(20, u_base_hashtag, u_time, relevant_docs_okapibm25_yake, docs_preprocessed)
# print('Mean Average Recall YAKE Model (Okapi BM25) : {}'.format(mean_average_recall_hashtag_okapibm25_yake))

# nDCG_hashtag_okapibm25_yake = nDCG(u_base_hashtag, u_time, relevant_docs_okapibm25_yake, docs_preprocessed)
# print('nDCG YAKE Model (Okapi BM25) : {}'.format(nDCG_hashtag_okapibm25_yake))

# """# Model with RAKE (Okapi BM25)"""

# relevant_docs_okapibm25_rake = find_relevant_documents_okapibm25(docs_preprocessed, tweet_keywords_rake)

# for rank, doc in enumerate(relevant_docs_okapibm25_rake):
#   print('Rank: {} Relevant Document: {}'.format(rank+1,doc))

# print()

# mean_average_precision_hashtag_okapibm25_rake = mean_average_precision(20, u_base_hashtag, u_time, relevant_docs_okapibm25_rake, docs_preprocessed)
# print('Mean Average Precision RAKE Model (Okapi BM25) : {}'.format(mean_average_precision_hashtag_okapibm25_rake))

# mean_average_recall_hashtag_okapibm25_rake = mean_average_recall(20, u_base_hashtag, u_time, relevant_docs_okapibm25_rake, docs_preprocessed)
# print('Mean Average Recall RAKE Model (Okapi BM25) : {}'.format(mean_average_recall_hashtag_okapibm25_rake))

# nDCG_hashtag_okapibm25_rake = nDCG(u_base_hashtag, u_time, relevant_docs_okapibm25_rake, docs_preprocessed)
# print('nDCG RAKE Model (Okapi BM25) : {}'.format(nDCG_hashtag_okapibm25_rake))

# """# Model with TextRank (Okapi BM25)"""

# relevant_docs_okapibm25_text_rank = find_relevant_documents_okapibm25(docs_preprocessed, tweet_keywords_text_rank)

# for rank, doc in enumerate(relevant_docs_okapibm25_text_rank):
#   print('Rank: {} Relevant Document: {}'.format(rank+1,doc))

# print()

# mean_average_precision_hashtag_okapibm25_text_rank = mean_average_precision(20, u_base_hashtag, u_time, relevant_docs_okapibm25_text_rank, docs_preprocessed)
# print('Mean Average Precision TextRank Model (Okapi BM25) : {}'.format(mean_average_precision_hashtag_okapibm25_text_rank))

# mean_average_recall_hashtag_okapibm25_text_rank = mean_average_recall(20, u_base_hashtag, u_time, relevant_docs_okapibm25_text_rank, docs_preprocessed)
# print('Mean Average Recall TextRank Model (Okapi BM25) : {}'.format(mean_average_recall_hashtag_okapibm25_text_rank))

# nDCG_hashtag_okapibm25_text_rank = nDCG(u_base_hashtag, u_time, relevant_docs_okapibm25_text_rank, docs_preprocessed)
# print('nDCG TextRank Model (Okapi BM25) : {}'.format(nDCG_hashtag_okapibm25_text_rank))

# """# Model with KeyBERT (Okapi BM25)"""

# # relevant_docs_okapibm25_keybert = find_relevant_documents_okapibm25(docs_preprocessed, tweet_keywords_keybert)

# # for rank, doc in enumerate(relevant_docs_okapibm25_keybert):
# #   print('Rank: {} Relevant Document: {}'.format(rank+1,doc))

# # print()

# # mean_average_precision_hashtag_okapibm25_keybert = mean_average_precision(20, u_base_hashtag, u_time, relevant_docs_okapibm25_keybert, docs_preprocessed)
# # print('Mean Average Precision KeyBERT Model (Okapi BM25) : {}'.format(mean_average_precision_hashtag_okapibm25_keybert))

# # mean_average_recall_hashtag_okapibm25_keybert = mean_average_recall(20, u_base_hashtag, u_time, relevant_docs_okapibm25_keybert, docs_preprocessed)
# # print('Mean Average Recall KeyBERT Model (Okapi BM25) : {}'.format(mean_average_recall_hashtag_okapibm25_keybert))

# # nDCG_hashtag_okapibm25_keybert = nDCG(u_base_hashtag, u_time, relevant_docs_okapibm25_keybert, docs_preprocessed)
# # print('nDCG KeyBERT Model (Okapi BM25) : {}'.format(nDCG_hashtag_okapibm25_keybert))

# """# Across all hashtags"""

# # global_list = [['narendramodi', '2022-02-14', 'india'],['UkraineRussiaCrisis', '2022-02-14', 'ukraine'],['IPL', '2022-02-14', 'india'],['TaylorSwift', '2022-02-14', 'USA'],['IndiaFightsCorona', '2022-02-14', 'india'],['narendramodi', '2022-02-15', 'india'],['UkraineRussiaCrisis', '2022-02-15', 'ukraine'],['IPL', '2022-02-15', 'india'],['TaylorSwift', '2022-02-15', 'USA'],['IndiaFightsCorona', '2022-02-15', 'india'],['narendramodi', '2022-02-16', 'india'],['UkraineRussiaCrisis', '2022-02-16', 'ukraine'],['IPL', '2022-02-16', 'india'],['TaylorSwift', '2022-02-16', 'USA'],['IndiaFightsCorona', '2022-02-16', 'india'],['narendramodi', '2022-02-17', 'india'],['UkraineRussiaCrisis', '2022-02-17', 'ukraine'],['IPL', '2022-02-17', 'india'],['TaylorSwift', '2022-02-17', 'USA'],['IndiaFightsCorona', '2022-02-17', 'india'],['narendramodi', '2022-02-18', 'india'],['UkraineRussiaCrisis', '2022-02-18', 'ukraine'],['IPL', '2022-02-18', 'india'],['TaylorSwift', '2022-02-18', 'USA'],['IndiaFightsCorona', '2022-02-18', 'india'],['narendramodi', '2022-02-19', 'india'],['UkraineRussiaCrisis', '2022-02-19', 'ukraine'],['IPL', '2022-02-19', 'india'],['hijab', '2022-02-19', 'india'],['vaccine', '2022-02-19', 'india'],['MillionAtIndiaPavilion', '2022-02-14', 'UAE'],['PunjabPanjeNaal', '2022-02-14', 'India'],['Euphoria', '2022-02-14', 'World'],['OscarsFanFavorite', '2022-02-14', 'World'],['ShameOnBirenSingh', '2022-02-14', 'india'],['BappiLahiri', '2022-02-16', 'india'],['BlandDoritos', '2022-02-16', 'USA'],['VERZUZ', '2022-02-16', 'USA'],['DragRaceUK', '2022-02-16', 'United Kingdom'],['BoycottWalgreens', '2022-02-18', 'USA'],['PunjabElections2022', '2022-02-20', 'india'],['WriddhimanSaha', '2022-02-20', 'india'],['stormfranklin', '2022-02-20', 'USA'],['QueenElizabeth', '2022-02-20', 'United Kingdom'],['ScottyFromWelding', '2022-02-20', 'Australia'],['CarabaoCupFinal', '2022-02-27', 'London'],['NZvSA', '2022-02-28', 'New Zealand'],['IPCC', '2022-02-28', 'Worldwide'],['SuperBowl', '2022-02-14', 'USA'],['MultiverseOfMadness', '2022-02-14', 'USA'],['Eminem', '2022-02-14', 'USA'],['IPLAuction', '2022-02-14', 'india'],['JohnsonOut21', '2022-02-14', 'United Kingdom'],['Cyberpunk2077', '2022-02-15', 'Worldwide'],['Wordle242', '2022-02-15', 'Worldwide'],['DeepSidhu', '2022-02-15', 'india'],['CanadaHasFallen', '2022-02-15', 'canada'],['IStandWithTrudeau', '2022-02-15', 'canada'],['CNNPHVPDebate', '2022-02-26', 'philippines'],['qldfloods', '2022-02-26', 'australia'],['Eurovision', '2022-02-26', 'worldwide'],['IndiansInUkraine', '2022-02-26', 'india'],['PritiPatel', '2022-02-26', 'united kingdom'],['TaylorCatterall', '2022-02-27', 'united kingdom'],['PSLFinal', '2022-02-27', 'pakistan'],['AustraliaDecides', '2022-02-27', 'australia'],['WorldNGODay', '2022-02-27', 'worldwide'],['TheBatman', '2022-02-28', 'USA'],['NationalScienceDay', '2022-02-28', 'india'],['msdtrong', '2022-02-14', 'india'],['Boycott_ChennaiSuperKings', '2022-02-14', 'india'],['GlanceJio', '2022-02-14', 'india'],['ArabicKuthu', '2022-02-14', 'india'],['Djokovic', '2022-02-15', 'australia'],['Real Madrid', '2022-02-15', 'santiago'],['bighit', '2022-02-15', 'korea'],['Maxwell', '2022-02-15', 'australia'],['mafsau', '2022-02-16', 'australia'],['channi', '2022-02-16', 'punjab'],['ayalaan', '2022-02-16', 'india'],['jkbose', '2022-02-16', 'india'],['HappyBirthdayPrinceSK', '2022-02-16', 'india'],['RandomActsOfKindnessDay', '2022-02-17', 'worldwide'],['happybirthdayjhope', '2022-02-17', 'korea'],['mohsinbaig', '2022-02-17', 'pakistan'],['aewdynamite', '2022-02-17', 'worldwide'],['aaraattu', '2022-02-17', 'india'],['ShivajiJayanti', '2022-02-18', 'india'],['PlotToKillModi', '2022-02-18', 'india'],['NationalDrinkWineDay', '2022-02-18', 'usa'],['HorizonForbiddenWest', '2022-02-18', 'worldwide'],['BoycottWalgreens', '2022-02-18', 'usa'],['CallTheMidwife', '2022-02-20', 'worldwide'],['OperationDudula', '2022-02-20', 'south africa'],['truthsocial', '2022-02-21', 'usa'],['nbaallstar', '2022-02-21', 'usa'],['shivamogga', '2022-02-21', 'india'],['HalftimeShow', '2022-02-14', 'usa'],['OttawaStrong', '2022-02-14', 'canada'],['DrDre', '2022-02-14', 'usa'],['BattleOfBillingsBridge', '2022-02-14', 'usa'],['FullyFaltooNFTdrop', '2022-02-14', 'worldwide'],['AK61', '2022-02-15', 'india'],['sandhyamukherjee', '2022-02-15', 'india'],['MUNBHA', '2022-02-15', 'worldwide'],['nursesstrike', '2022-02-15', 'australia'],['Realme9ProPlus', '2022-02-16', 'worldwide'],['KarnatakaHijabControversy', '2022-02-16', 'india'],['BJPwinningUP', '2022-02-16', 'india'],['Punjab_With_Modi', '2022-02-16', 'india'],['PushpaTheRule', '2022-02-16', 'india'],['RehmanMalik', '2022-02-22', 'india'],['harisrauf', '2022-02-22', 'pakistan'],['Rosettenville', '2022-02-22', 'south africa'],['NFU22', '2022-02-22', 'worldwide'],['justiceforharsha', '2022-02-22', 'india'],['wordle251', '2022-02-24', 'worldwide'],['ARSWOL', '2022-02-24', 'worldwide'],['stopwar', '2022-02-24', 'worldwide'],['PrayForPeace', '2022-02-24', 'worldwide'],['StopPutinNOW', '2022-02-24', 'worldwide'],['TeamGirlsCup', '2022-02-25', 'worldwide'],['Canucks', '2022-02-25', 'worldwide'],['PinkShirtDay', '2022-02-25', 'canada'],['superrugbypacific', '2022-02-25', 'australia']]

# # global_average_mean_average_precision = []
# # global_mean_average_recall = []

# # for iter in tqdm(range(len(global_list))):
# #   u_base_hashtag = global_list[iter][0]
# #   u_time = global_list[iter][1]
# #   u_location = global_list[iter][2]
# #   tweet_query = []
# #   format = '%Y-%m-%d'
# #   u_present_date = datetime.datetime.strptime(u_time, format)
# #   u_prev_date = u_present_date - datetime.timedelta(days=1)
# #   u_next_date = u_present_date + datetime.timedelta(days=1)
# #   df_query = df.loc[df['hashtags'].str.contains(u_base_hashtag) & df['Date_Only'].isin([str(u_present_date.date()), str(u_prev_date.date()), str(u_next_date.date())])]

# #   for tweet in df_query['Preprocessed_Data']:
# #     tweet_query.extend(tweet)
  
# #   tweet_keywords = []
# #   kw_extractor = yake.KeywordExtractor(top=20, stopwords=None)
# #   keywords = kw_extractor.extract_keywords(' '.join(tweet_query))
# #   for kw, v in keywords:
# #     #print("Keyphrase: ",kw, ": score", v)
# #     for key in kw.split():
# #       if(key not in tweet_keywords):
# #         tweet_keywords.append(key)
  
# #   docs_preprocessed = []

# #   total_documents = 0
# #   path = '/content/drive/MyDrive/Tweelink_Dataset/Tweelink_Articles_Processed'
# #   for filename in glob(os.path.join(path, '*')):
# #     with open(os.path.join(os.getcwd(), filename), 'r', encoding = 'utf-8',errors = 'ignore') as f:
# #       filename = os.path.basename(f.name)
# #       data = json.load(f)
# #       d_date = data["Date"]
# #       if(d_date=="" or d_date=="Date"):
# #         continue
# #       format = '%Y-%m-%d'
  
# #       d_present_date = datetime.datetime.strptime(d_date, format)
  
# #       if(str(d_present_date.date()) not in [str(u_present_date.date()), str(u_prev_date.date()), str(u_next_date.date())]):
# #         continue
    
# #       docs_preprocessed.append({'Name':filename, 'Data':data})
# #       total_documents+=1
  
# #   # with yake
# #   # relevant_docs = find_relevant_documents(docs_preprocessed, tweet_keywords)

#   # without yake/ keyword extraction
#   relevant_docs = find_relevant_documents(docs_preprocessed, tweet_query)

#   mean_average_precision_hashtag = mean_average_precision(20, u_base_hashtag, u_time, relevant_docs, docs_preprocessed)
#   global_average_mean_average_precision.append(mean_average_precision_hashtag)
#   #print(mean_average_precision_hashtag)

#   mean_average_recall_hashtag = mean_average_recall(20, u_base_hashtag, u_time, relevant_docs, docs_preprocessed)
#   global_mean_average_recall.append(mean_average_recall_hashtag)
#   #print(mean_average_recall_hashtag)

# overall_average_mean_average_precision = sum(global_average_mean_average_precision)/len(global_average_mean_average_precision)
# print(overall_average_mean_average_precision)

# overall_mean_average_recall = sum(global_mean_average_recall)/len(global_mean_average_recall)
# print(overall_mean_average_recall)