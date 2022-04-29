#!/usr/bin/env python
# coding: utf-8

# # Import Statements

# In[1]:


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
get_ipython().system(' pip install multi_rake')
from multi_rake import Rake
get_ipython().system(' pip install summa')
from summa import keywords as summa_keywords
get_ipython().system(' pip install keybert')
from keybert import KeyBERT


# In[2]:


from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')


# In[3]:


from google.colab import drive
drive.mount('/content/drive')


# In[4]:


file1 = open("/content/drive/MyDrive/Tweelink_Dataset/twitter_base_preprocessed.pkl", "rb")
df = pickle.load(file1)
file1.close()


# # Input

# In[6]:


u_base_hashtag = input("Enter base hashtag: ")
u_time = input("Enter time: ")
u_location = input("Enter Location: ")


# # Processing

# In[7]:


import datetime
tweet_query = []
format = '%Y-%m-%d'
u_present_date = datetime.datetime.strptime(u_time, format)
u_prev_date = u_present_date - datetime.timedelta(days=1)
u_next_date = u_present_date + datetime.timedelta(days=1)
df_query = df.loc[df['hashtags'].str.contains(u_base_hashtag) & df['Date_Only'].isin([str(u_present_date.date()), str(u_prev_date.date()), str(u_next_date.date())])]
# print(df_query.shape[0])
if df_query.shape[0]<50:
  df_query = df.loc[df['Date_Only'].isin([str(u_present_date.date()), str(u_prev_date.date()), str(u_next_date.date())])]
  df_query = df.iloc[:min(df_query.shape[0],1000),:]


# In[8]:


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

  if len(preprocessed_vocabulary) > 0:
    AOF_coefficient = sum(preprocessed_vocabulary.values())/len(preprocessed_vocabulary)
  else:
    AOF_coefficient = 0
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


# In[9]:


for tweet in df_query['Preprocessed_Data']:
  tweet_query.extend(tweet)


# In[10]:


def get_keyword_keywordextractor(df_query):
  keyword_dataset = df_query['tweet'].tolist()
  tweet_query_keyword_extractor = keyword_extractor(keyword_dataset)
  return tweet_query_keyword_extractor


# In[11]:


def get_yake_keywords(tweet_query):
  tweet_keywords_yake = []
  kw_extractor = yake.KeywordExtractor(top=20, stopwords=None)
  keywords = kw_extractor.extract_keywords(' '.join(tweet_query))
  #keywords = kw_extractor.extract_keywords(' '.join(df_query['tweet'].tolist()))
  for kw, v in keywords:
    # print("Keyphrase: ",kw, ": score", v)
    for key in kw.split():
      if(key.lower() not in tweet_keywords_yake):
        tweet_keywords_yake.append(key.lower())
  return tweet_keywords_yake


# In[12]:


def get_rake_keywords(tweet_query):
  tweet_keywords_rake = []
  rake = Rake()
  rake_keywords = rake.apply(' '.join(tweet_query).encode('ascii', 'ignore').decode())
  for kw,score in rake_keywords[:20]:
    for key in kw.split():
      if(key.lower() not in tweet_keywords_rake):
        tweet_keywords_rake.append(key.lower())
  # print(tweet_keywords_rake)
  return tweet_keywords_rake


# In[13]:


def get_text_rank_keywords(tweet_query):
  tweet_keywords_text_rank = []
  TR_keywords = summa_keywords.keywords(' '.join(tweet_query), scores=True)
  for kw,score in TR_keywords[:20]:
    for key in kw.split():
      if(key.lower() not in tweet_keywords_text_rank):
        tweet_keywords_text_rank.append(key.lower())
  # print(tweet_keywords_text_rank)
  return tweet_keywords_text_rank


# In[14]:


def get_keybert_keywords(tweet_query):
  # uncomment later
  keybert_model = KeyBERT(model='all-mpnet-base-v2')
  keybert_keywords = keybert_model.extract_keywords(' '.join(tweet_query), keyphrase_ngram_range=(1,1), stop_words='english', highlight=False, top_n=20)
  tweet_keywords_keybert = list(dict(keybert_keywords).keys())
  # print(tweet_keywords_keybert)
  return tweet_keywords_keybert


# # Helpful Functions

# In[15]:


docs_preprocessed = []


# In[16]:


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
 
     try:
       d_present_date = datetime.datetime.strptime(d_date, format)
     except:
       continue
 
     if(str(d_present_date.date()) not in [str(u_present_date.date()), str(u_prev_date.date()), str(u_next_date.date())]):
       continue
   
     docs_preprocessed.append({'Name':filename, 'Data':data})
     total_documents+=1
# print(total_documents)


# In[17]:


def get_relevant_docs_list_for_base_hashtag(base_hashtag, base_date, docs_preprocessed):
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


# In[18]:


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


# In[19]:


def precision_at_k(k, base_hashtag, base_date, prediction_list, docs_preprocessed):
  relevant_docs_list = get_relevant_docs_list_for_base_hashtag(base_hashtag, base_date, docs_preprocessed)
  num_of_relevant_results=0
  list_of_rel_doc_names = []
  for x in relevant_docs_list:
    list_of_rel_doc_names.append(x[0])
  for itr in range(min(len(prediction_list), k)):
    if (prediction_list[itr][0] in list_of_rel_doc_names):
      num_of_relevant_results+=1
  return num_of_relevant_results/k


# In[20]:


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


# In[21]:


def recall_at_k(k, base_hashtag, base_date, prediction_list, docs_preprocessed):
  relevant_docs_list = get_relevant_docs_list_for_base_hashtag(base_hashtag, base_date, docs_preprocessed)
  current_num_of_relevant_results=0
  list_of_rel_doc_names = []
  for x in relevant_docs_list:
    list_of_rel_doc_names.append(x[0])
  for itr in range(min(len(prediction_list), k)):
    if (prediction_list[itr][0] in list_of_rel_doc_names):
      current_num_of_relevant_results+=1
  if(len(relevant_docs_list)==0):
    return 0
  return current_num_of_relevant_results/len(relevant_docs_list)


# In[22]:


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


# # TF-IDF Binary

# In[23]:


#Finding vocabulary of our dataset
all_words = []
for doc in docs_preprocessed:
  #print(doc)
  tokens = doc['Data']['Body_processed']
  all_words.extend(tokens)

# finding all distinct words in dataset
vocabulary = list(set(all_words))
word2id = {}
id2word = {}

k = 0
for word in vocabulary: 
  word2id[word] = k
  k += 1
for word in word2id.keys():
  id2word[word2id[word]] = word


# In[24]:


# Word-doc matrix (for storing frequency of each word of vocabulary in document)
inverted_index = {}
total_words_in_doc = []
word_doc_matrix = np.zeros((total_documents, len(vocabulary)), dtype=float)
for i in tqdm(range(len(docs_preprocessed))):
  # total_words_in_doc.append(len(docs_preprocessed[i]['Text']))
  for j, word in enumerate(docs_preprocessed[i]['Data']['Body_processed']):
    if(word not in inverted_index):
        inverted_index[word] = [i]
    else:
      if inverted_index[word][-1]!=i:
        inverted_index[word].append(i)
    word_doc_matrix[i][word2id[word]] += 1
  total_words_in_doc.append(sum(word_doc_matrix[i]))


# In[25]:


max_freq_word_in_doc = []
for words_in_doc in word_doc_matrix:
  max_freq_word_in_doc.append(np.max(words_in_doc))


# In[26]:


IDF = {}
inverted_index_final = {}
for key in inverted_index:
  inverted_index_final[key] = {"Frequency": len(inverted_index[key]), "doc_id":inverted_index[key]}
  IDF[key] = np.log10(total_documents/(len(inverted_index[key]) +1))

#  sorting according to words
inverted_index_final = dict(sorted(inverted_index_final.items()))


# In[27]:


def Binary_tf_idf_scheme(word_doc_matrix):
  return np.where((word_doc_matrix > 0),1,0).astype('float')


# In[28]:


Binary_tf_idf = Binary_tf_idf_scheme(np.copy(word_doc_matrix))


# In[29]:


for i in tqdm(range(len(vocabulary))):
  IDF_factor = IDF[id2word[i]]
  Binary_tf_idf[:,i] *= IDF_factor


# In[30]:


def find_relevant_documents_tfidf_binary(isPlain, docs_preprocessed, processed_query):
  max_list_size = len(get_relevant_docs_list_for_base_hashtag(u_base_hashtag, u_time, docs_preprocessed))
  
  query_frequency = dict()
  for query_token in processed_query:
      if query_token in query_frequency:
          query_frequency[query_token] += 1
      else:
        query_frequency[query_token] = 1

  query_frequency_vector = np.zeros((len(vocabulary),))

  for token in query_frequency.keys():
    if token in vocabulary:
      query_frequency_vector[word2id[token]] = query_frequency[token]

  max_freq_token = query_frequency[max(query_frequency, key=query_frequency.get)]
  total_tokens = len(processed_query)

  f = query_frequency_vector
  query_binary = Binary_tf_idf_scheme(np.copy(query_frequency_vector))

  for i in tqdm(range(len(vocabulary))):
    IDF_factor = IDF[id2word[i]]
    query_binary[i] *= IDF_factor

  if isPlain:
    query_binary = np.where(query_binary >= np.sort(query_binary)[-10], query_binary, 0)
  
  Binary_tf_idf_score = np.dot(Binary_tf_idf, query_binary)
  Binary_tf_idf_score = {docs_preprocessed[i]['Name']:Binary_tf_idf_score[i] for i in range(len(Binary_tf_idf_score))}
  Binary_relevant_docs = list( sorted(Binary_tf_idf_score.items(), key=operator.itemgetter(1),reverse=True))[:max_list_size]

  for i in range(len(Binary_relevant_docs)):
    for j in range(len(docs_preprocessed)):
      if(Binary_relevant_docs[i][0] == docs_preprocessed[j]['Name']):
        Binary_relevant_docs[i] = (Binary_relevant_docs[i][0], Binary_relevant_docs[i][1], docs_preprocessed[j]['Data']['Date'], docs_preprocessed[j]['Data']['Location'].lower(), docs_preprocessed[j]['Data']['Link'])

  # prioritize location
  location_relevant = []
  location_irrelevant = []
  for x in Binary_relevant_docs:
    if u_location.lower() in x[3]:
      location_relevant.append(x)
    else:
      location_irrelevant.append(x) 
  Binary_relevant_docs = location_relevant + location_irrelevant
  return Binary_relevant_docs


# # Plain Model (TF-IDF Binary)

# In[31]:


relevant_docs_tfidf_binary_plain = find_relevant_documents_tfidf_binary(True, docs_preprocessed, tweet_query)

for rank, doc in enumerate(relevant_docs_tfidf_binary_plain):
  print('Rank: {} Relevant Document: {}'.format(rank+1,doc))

print()

mean_average_precision_hashtag_tfidf_binary_plain = mean_average_precision(20, u_base_hashtag, u_time, relevant_docs_tfidf_binary_plain, docs_preprocessed)
print('Mean Average Precision Plain Model (TF-IDF Binary) : {}'.format(mean_average_precision_hashtag_tfidf_binary_plain))

mean_average_recall_hashtag_tfidf_binary_plain = mean_average_recall(20, u_base_hashtag, u_time, relevant_docs_tfidf_binary_plain, docs_preprocessed)
print('Mean Average Recall Plain Model (TF-IDF Binary) : {}'.format(mean_average_recall_hashtag_tfidf_binary_plain))

nDCG_hashtag_tfidf_binary_plain = nDCG(u_base_hashtag, u_time, relevant_docs_tfidf_binary_plain, docs_preprocessed)
print('nDCG Plain Model (TF-IDF Binary) : {}'.format(nDCG_hashtag_tfidf_binary_plain))


# # Model with Keyword Extractor (TF-IDF Binary)

# In[32]:


tweet_query_keyword_extractor = get_keyword_keywordextractor(df_query)
relevant_docs_tfidf_binary_keyword_extractor = find_relevant_documents_tfidf_binary(False, docs_preprocessed, tweet_query_keyword_extractor[:20])

for rank, doc in enumerate(relevant_docs_tfidf_binary_keyword_extractor):
  print('Rank: {} Relevant Document: {}'.format(rank+1,doc))

print()

mean_average_precision_hashtag_tfidf_binary_keyword_extractor = mean_average_precision(20, u_base_hashtag, u_time, relevant_docs_tfidf_binary_keyword_extractor, docs_preprocessed)
print('Mean Average Precision Keyword Extractor Model (TF-IDF Binary) : {}'.format(mean_average_precision_hashtag_tfidf_binary_keyword_extractor))

mean_average_recall_hashtag_tfidf_binary_keyword_extractor = mean_average_recall(20, u_base_hashtag, u_time, relevant_docs_tfidf_binary_keyword_extractor, docs_preprocessed)
print('Mean Average Recall Keyword Extractor Model (TF-IDF Binary) : {}'.format(mean_average_recall_hashtag_tfidf_binary_keyword_extractor))

nDCG_hashtag_tfidf_binary_keyword_extractor = nDCG(u_base_hashtag, u_time, relevant_docs_tfidf_binary_keyword_extractor, docs_preprocessed)
print('nDCG Keyword Extractor Model (TF-IDF Binary) : {}'.format(nDCG_hashtag_tfidf_binary_keyword_extractor))


# # Model with YAKE (TF-IDF Binary)

# In[33]:


# Model with YAKE
tweet_keywords_yake = get_yake_keywords(tweet_query)
relevant_docs_tfidf_binary_yake = find_relevant_documents_tfidf_binary(False, docs_preprocessed, tweet_keywords_yake)

for rank, doc in enumerate(relevant_docs_tfidf_binary_yake):
  print('Rank: {} Relevant Document: {}'.format(rank+1,doc))

print()

mean_average_precision_hashtag_tfidf_binary_yake = mean_average_precision(20, u_base_hashtag, u_time, relevant_docs_tfidf_binary_yake, docs_preprocessed)
print('Mean Average Precision Keyword YAKE Model (TF-IDF Binary) : {}'.format(mean_average_precision_hashtag_tfidf_binary_yake))

mean_average_recall_hashtag_tfidf_binary_yake = mean_average_recall(20, u_base_hashtag, u_time, relevant_docs_tfidf_binary_yake, docs_preprocessed)
print('Mean Average Recall Keyword YAKE Model (TF-IDF Binary) : {}'.format(mean_average_recall_hashtag_tfidf_binary_yake))

nDCG_hashtag_tfidf_binary_yake = nDCG(u_base_hashtag, u_time, relevant_docs_tfidf_binary_yake, docs_preprocessed)
print('nDCG Keyword YAKE Model (TF-IDF Binary) : {}'.format(nDCG_hashtag_tfidf_binary_yake))


# # Model with RAKE (TF-IDF Binary)

# In[34]:


# Model with RAKE
tweet_keywords_rake = get_rake_keywords(tweet_query)
relevant_docs_tfidf_binary_rake = find_relevant_documents_tfidf_binary(False, docs_preprocessed, tweet_keywords_rake)

for rank, doc in enumerate(relevant_docs_tfidf_binary_rake):
  print('Rank: {} Relevant Document: {}'.format(rank+1,doc))

print()

mean_average_precision_hashtag_tfidf_binary_rake = mean_average_precision(20, u_base_hashtag, u_time, relevant_docs_tfidf_binary_rake, docs_preprocessed)
print('Mean Average Precision Keyword RAKE Model (TF-IDF Binary) : {}'.format(mean_average_precision_hashtag_tfidf_binary_rake))

mean_average_recall_hashtag_tfidf_binary_rake = mean_average_recall(20, u_base_hashtag, u_time, relevant_docs_tfidf_binary_rake, docs_preprocessed)
print('Mean Average Recall Keyword RAKE Model (TF-IDF Binary) : {}'.format(mean_average_recall_hashtag_tfidf_binary_rake))

nDCG_hashtag_tfidf_binary_rake = nDCG(u_base_hashtag, u_time, relevant_docs_tfidf_binary_rake, docs_preprocessed)
print('nDCG Keyword RAKE Model (TF-IDF Binary) : {}'.format(nDCG_hashtag_tfidf_binary_rake))


# # Model with TextRank (TF-IDF Binary)

# In[35]:


# Model with TextRank
tweet_keywords_text_rank = get_text_rank_keywords(tweet_query)
relevant_docs_tfidf_binary_text_rank = find_relevant_documents_tfidf_binary(False, docs_preprocessed, tweet_keywords_text_rank)

for rank, doc in enumerate(relevant_docs_tfidf_binary_text_rank):
  print('Rank: {} Relevant Document: {}'.format(rank+1,doc))

print()

mean_average_precision_hashtag_tfidf_binary_text_rank = mean_average_precision(20, u_base_hashtag, u_time, relevant_docs_tfidf_binary_text_rank, docs_preprocessed)
print('Mean Average Precision Keyword TextRank Model (TF-IDF Binary) : {}'.format(mean_average_precision_hashtag_tfidf_binary_text_rank))

mean_average_recall_hashtag_tfidf_binary_text_rank = mean_average_recall(20, u_base_hashtag, u_time, relevant_docs_tfidf_binary_text_rank, docs_preprocessed)
print('Mean Average Recall Keyword TextRank Model (TF-IDF Binary) : {}'.format(mean_average_recall_hashtag_tfidf_binary_text_rank))

nDCG_hashtag_tfidf_binary_text_rank = nDCG(u_base_hashtag, u_time, relevant_docs_tfidf_binary_text_rank, docs_preprocessed)
print('nDCG Keyword TextRank Model (TF-IDF Binary) : {}'.format(nDCG_hashtag_tfidf_binary_text_rank))


# # Model with KeyBERT (TF-IDF Binary)

# In[36]:


# uncomment later
# Model with KeyBERT
tweet_keywords_keybert = get_keybert_keywords(tweet_query)
relevant_docs_tfidf_binary_keybert = find_relevant_documents_tfidf_binary(False, docs_preprocessed, tweet_keywords_keybert)

for rank, doc in enumerate(relevant_docs_tfidf_binary_keybert):
  print('Rank: {} Relevant Document: {}'.format(rank+1,doc))

print()

mean_average_precision_hashtag_tfidf_binary_keybert = mean_average_precision(20, u_base_hashtag, u_time, relevant_docs_tfidf_binary_keybert, docs_preprocessed)
print('Mean Average Precision Keyword KeyBERT Model (TF-IDF Binary) : {}'.format(mean_average_precision_hashtag_tfidf_binary_keybert))

mean_average_recall_hashtag_tfidf_binary_keybert = mean_average_recall(20, u_base_hashtag, u_time, relevant_docs_tfidf_binary_keybert, docs_preprocessed)
print('Mean Average Recall Keyword KeyBERT Model (TF-IDF Binary) : {}'.format(mean_average_recall_hashtag_tfidf_binary_keybert))

nDCG_hashtag_tfidf_binary_keybert = nDCG(u_base_hashtag, u_time, relevant_docs_tfidf_binary_keybert, docs_preprocessed)
print('nDCG Keyword KeyBERT Model (TF-IDF Binary) : {}'.format(nDCG_hashtag_tfidf_binary_keybert))


# In[37]:


global_list =[['IPL', '2022-02-25', 'india'], ['IPL', '2022-02-16', 'india'], ['Drishyam2', '2022-02-17', 'india'], ['elections', '2022-02-27', 'imphal'], ['elections', '2022-02-27', 'new delhi'], ['elections', '2022-02-27', 'uttar pradesh'], ['nursesstrike', '2022-02-15', 'australia'], ['Djokovic', '2022-02-15', 'australia'],['hijab', '2022-02-19', 'india'], ['msdtrong', '2022-02-14', 'Florida'], ['ottawaoccupied', '2022-02-14', 'Ottawa'], ['Boycott_ChennaiSuperKings', '2022-02-14', 'chennai'], ['GlanceJio', '2022-02-14', 'New Delhi'], ['ArabicKuthu', '2022-02-14', 'CHENNAI'], ['Djokovic', '2022-02-15', 'AUSTRALIA'], ['Real Madrid', '2022-02-15', 'Madrid'], ['bighit', '2022-02-15', 'ASSAM'], ['Maxwell', '2022-02-15', 'india'], ['mafsau', '2022-02-16', 'australia'], ['channi', '2022-02-16', 'punjab'], ['ayalaan', '2022-02-16', 'tamil nadu'], ['jkbose', '2022-02-16', 'srinagar'], ['happybirthdayjhope', '2022-02-17', 'new delhi'], ['mohsinbaig', '2022-02-17', 'islamabad'],  ['ShivajiJayanti', '2022-02-18', 'maharashtra'], ['OperationDudula', '2022-02-20', 'South Africa'], ['UFCVegas48', '2022-02-20', 'india'], ['FCNPSG', '2022-02-20', 'PARIS'], ['shivamogga', '2022-02-21', 'KARNATAKA'],  ['StayAlive_CHAKHO', '2022-02-22', 'seoul'], ['KaranSinghGrover', '2022-02-22', 'india'], ['NationalMargaritaDay', '2022-02-22', 'United States of America'], ['dontsaygay', '2022-02-22', 'New York'], ['RIPRikyRick', '2022-02-23', 'South Africa'], ['budget2022', '2022-02-23', 'Cape Town'], ['CottonFest2022', '2022-02-23', 'nigeria'], ['NationalChiliDay', '2022-02-24', 'Ohio'], ['stockmarketcrash', '2022-02-24', 'mumbai'], ['BidenisaFailure', '2022-02-24', 'United States of America']]
print(len(global_list))
global_list = global_list[:30]


# In[ ]:


# global_average_mean_average_precision_plain = []
# global_mean_average_recall_plain = []

# global_average_mean_average_precision_keyword_extractor = []
# global_mean_average_recall_keyword_extractor = []

# global_average_mean_average_precision_rake = []
# global_mean_average_recall_rake = []

# global_average_mean_average_precision_yake = []
# global_mean_average_recall_yake = []

# global_average_mean_average_precision_text_rank = []
# global_mean_average_recall_text_rank = []


# global_average_mean_average_precision_keybert = []
# global_mean_average_recall_keybert = []

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
  
#   tweet_keywords = []
#   kw_extractor = yake.KeywordExtractor(top=20, stopwords=None)
#   keywords = kw_extractor.extract_keywords(' '.join(tweet_query))
#   for kw, v in keywords:
#     #print("Keyphrase: ",kw, ": score", v)
#     for key in kw.split():
#       if(key not in tweet_keywords):
#         tweet_keywords.append(key)
  
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
  
#   # Plain
#   relevant_articles_list_plain = find_relevant_documents_tfidf_binary(True, docs_preprocessed, tweet_query)
#   mean_average_precision_hashtag_plain = mean_average_precision(20, u_base_hashtag, u_time, relevant_articles_list_plain, docs_preprocessed)
#   global_average_mean_average_precision_plain.append(mean_average_precision_hashtag_plain)
#   mean_average_recall_hashtag_plain = mean_average_recall(20, u_base_hashtag, u_time, relevant_articles_list_plain, docs_preprocessed)
#   global_mean_average_recall_plain.append(mean_average_recall_hashtag_plain)

#   # keyword extractor
#   tweet_query_keyword_extractor = get_keyword_keywordextractor(df_query)
#   relevant_articles_list_keyword_extractor = find_relevant_documents_tfidf_binary(False, docs_preprocessed, tweet_query_keyword_extractor[:20])
#   mean_average_precision_hashtag_keyword_extractor = mean_average_precision(20, u_base_hashtag, u_time, relevant_articles_list_keyword_extractor, docs_preprocessed)
#   global_average_mean_average_precision_keyword_extractor.append(mean_average_precision_hashtag_keyword_extractor)
#   mean_average_recall_hashtag_keyword_extractor = mean_average_recall(20, u_base_hashtag, u_time, relevant_articles_list_keyword_extractor, docs_preprocessed)
#   global_mean_average_recall_keyword_extractor.append(mean_average_recall_hashtag_keyword_extractor)

#   # Rake
#   tweet_query_rake = get_rake_keywords(tweet_query)
#   relevant_articles_list_rake = find_relevant_documents_tfidf_binary(False, docs_preprocessed, tweet_query_rake)
#   mean_average_precision_hashtag_rake = mean_average_precision(20, u_base_hashtag, u_time, relevant_articles_list_rake, docs_preprocessed)
#   global_average_mean_average_precision_rake.append(mean_average_precision_hashtag_rake)
#   mean_average_recall_hashtag_rake = mean_average_recall(20, u_base_hashtag, u_time, relevant_articles_list_rake, docs_preprocessed)
#   global_mean_average_recall_rake.append(mean_average_recall_hashtag_rake)


#    # Yake
#   tweet_query_yake = get_yake_keywords(tweet_query)
#   relevant_articles_list_yake = find_relevant_documents_tfidf_binary(False, docs_preprocessed, tweet_query_yake)
#   mean_average_precision_hashtag_yake = mean_average_precision(20, u_base_hashtag, u_time, relevant_articles_list_yake, docs_preprocessed)
#   global_average_mean_average_precision_yake.append(mean_average_precision_hashtag_yake)
#   mean_average_recall_hashtag_yake = mean_average_recall(20, u_base_hashtag, u_time, relevant_articles_list_yake, docs_preprocessed)
#   global_mean_average_recall_yake.append(mean_average_recall_hashtag_yake)


#   # Text Rank
#   tweet_query_text_rank = get_text_rank_keywords(tweet_query)
#   relevant_articles_list_text_rank = find_relevant_documents_tfidf_binary(False, docs_preprocessed, tweet_query_text_rank)
#   mean_average_precision_hashtag_text_rank = mean_average_precision(20, u_base_hashtag, u_time, relevant_articles_list_text_rank, docs_preprocessed)
#   global_average_mean_average_precision_text_rank.append(mean_average_precision_hashtag_text_rank)
#   mean_average_recall_hashtag_text_rank = mean_average_recall(20, u_base_hashtag, u_time, relevant_articles_list_text_rank, docs_preprocessed)
#   global_mean_average_recall_text_rank.append(mean_average_recall_hashtag_text_rank)


#   # Keybert
#   tweet_query_keybert = get_keybert_keywords(tweet_query)
#   relevant_articles_list_keybert = find_relevant_documents_tfidf_binary(False, docs_preprocessed, tweet_query_keybert)
#   mean_average_precision_hashtag_keybert = mean_average_precision(20, u_base_hashtag, u_time, relevant_articles_list_text_rank, docs_preprocessed)
#   global_average_mean_average_precision_keybert.append(mean_average_precision_hashtag_keybert)
#   mean_average_recall_hashtag_keybert = mean_average_recall(20, u_base_hashtag, u_time, relevant_articles_list_keybert, docs_preprocessed)
#   global_mean_average_recall_keybert.append(mean_average_recall_hashtag_keybert)


# In[ ]:


# overall_average_mean_average_precision_plain = sum(global_average_mean_average_precision_plain)/len(global_average_mean_average_precision_plain)
# print(overall_average_mean_average_precision_plain)

# overall_mean_average_recall_plain = sum(global_mean_average_recall_plain)/len(global_mean_average_recall_plain)
# print(overall_mean_average_recall_plain)


# In[ ]:


# overall_average_mean_average_precision_keyword_extractor = sum(global_average_mean_average_precision_keyword_extractor)/len(global_average_mean_average_precision_keyword_extractor)
# print(overall_average_mean_average_precision_keyword_extractor)

# overall_mean_average_recall_keyword_extractor = sum(global_mean_average_recall_keyword_extractor)/len(global_mean_average_recall_keyword_extractor)
# print(overall_mean_average_recall_keyword_extractor)


# In[ ]:


# overall_average_mean_average_precision_rake = sum(global_average_mean_average_precision_rake)/len(global_average_mean_average_precision_rake)
# print(overall_average_mean_average_precision_rake)

# overall_mean_average_recall_rake = sum(global_mean_average_recall_rake)/len(global_mean_average_recall_rake)
# print(overall_mean_average_recall_rake)


# In[ ]:


# overall_average_mean_average_precision_yake = sum(global_average_mean_average_precision_yake)/len(global_average_mean_average_precision_yake)
# print(overall_average_mean_average_precision_yake)

# overall_mean_average_recall_yake = sum(global_mean_average_recall_yake)/len(global_mean_average_recall_yake)
# print(overall_mean_average_recall_yake)


# In[ ]:



# overall_average_mean_average_precision_text_rank = sum(global_average_mean_average_precision_text_rank)/len(global_average_mean_average_precision_text_rank)
# print(overall_average_mean_average_precision_text_rank)

# overall_mean_average_recall_text_rank = sum(global_mean_average_recall_text_rank)/len(global_mean_average_recall_text_rank)
# print(overall_mean_average_recall_text_rank)


# In[ ]:


# overall_average_mean_average_precision_keybert = sum(global_average_mean_average_precision_keybert)/len(global_average_mean_average_precision_keybert)
# print(overall_average_mean_average_precision_keybert)

# overall_mean_average_recall_keybert = sum(global_mean_average_recall_keybert)/len(global_mean_average_recall_keybert)
# print(overall_mean_average_recall_keybert)

