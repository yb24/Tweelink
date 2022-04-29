#!/usr/bin/env python
# coding: utf-8

# # Import

# In[2]:


from collections import defaultdict
from math import log, sqrt
import re
import numpy as np
import sys
from copy import deepcopy
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
import datetime
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('corpus')
get_ipython().system(' pip install git+https://github.com/LIAAD/yake')
import yake
get_ipython().system(' pip install multi_rake')
from multi_rake import Rake
get_ipython().system(' pip install summa')
from summa import keywords as summa_keywords
get_ipython().system(' pip install keybert')
from keybert import KeyBERT


# In[3]:


from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')


# In[4]:


from google.colab import drive
drive.mount('/content/drive')


# In[5]:


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
  AOF_coefficient = 0
  if len(preprocessed_vocabulary) > 0:
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


# In[8]:


# ## Import the data and create the inverted index

def import_dataset():
    """
    This function import all the articles in the TIME corpus,
    returning list of lists where each sub-list contains all the
    terms present in the document as a string.
    """
    # articles = []
    # with open('TIME.ALL', 'r') as f:
    #     tmp = []
    #     for row in f:
    #         if row.startswith("*TEXT"):
    #             if tmp != []:
    #                 articles.append(tmp)
    #             tmp = []
    #         else:
    #             row = re.sub(r'[^a-zA-Z\s]+', '', row)
    #             tmp += row.split()
    # return articles

    docs_preprocessed_metrics = []
    docs_preprocessed = []
    docs_preprocessed_with_names = []
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
        docs_preprocessed_metrics.append({'Name':filename, 'Data':data})
        docs_preprocessed.append(data['Body_processed'])
        docs_preprocessed_with_names.append(filename)

    return docs_preprocessed, docs_preprocessed_with_names, docs_preprocessed_metrics




# def remove_stop_words(corpus):
#    '''
#    This function removes from the corpus all the stop words present in the file TIME.STP
#    '''
#    stop_w = [line.rstrip('\n') for line in open('TIME.STP')]
#    stop_w=list(filter(None, stop_w))
#    for i in range(0,len(corpus)):
#        corpus[i] = [x for x in corpus[i] if x not in stop_w]
#    return corpus 


def make_inverted_index(corpus):
    """
    This function builds an inverted index as an hash table (dictionary)
    where the keys are the terms and the values are ordered lists of
    docIDs containing the term.
    """
    # corpus = remove_stop_words(corpus)
    index = defaultdict(set)
    for docid, article in enumerate(corpus):
        for term in article:
            index[term].add(docid)
    return index


# ### Union of two posting lists


def posting_lists_union(pl1, pl2):
        """
        Returns a new posting list resulting from the union of the
        two lists passed as arguments.
        """
        pl1 = sorted(list(pl1))
        pl2 = sorted(list(pl2))
        union = []
        i = 0
        j = 0
        while (i < len(pl1) and j < len(pl2)):
            if (pl1[i] == pl2[j]):
                union.append(pl1[i])
                i += 1
                j += 1
            elif (pl1[i] < pl2[j]):
                union.append(pl1[i])
                i += 1
            else:
                union.append(pl2[j])
                j += 1
        for k in range(i, len(pl1)):
            union.append(pl1[k])
        for k in range(j, len(pl2)):
            union.append(pl2[k])
        return union


# ## Precomputing weights


def DF(term, index):
    '''
    Function computing Document Frequency for a term.
    '''
    return len(index[term])


def IDF(term, index, corpus):
    '''
    Function computing Inverse Document Frequency for a term.
    '''
    return log(len(corpus)/DF(term, index))


def RSV_weights(corpus,index):
    '''
    This function precomputes the Retrieval Status Value weights
    for each term in the index
    '''
    N = len(corpus)
    w = {}
    for term in index.keys():
        p = DF(term, index)/(N+0.5)  
        w[term] = IDF(term, index, corpus) + log(p/(1-p))
    return w
    


# ## BIM Class


class BIM():
    '''
    Binary Independence Model class
    '''
    
    def __init__(self, corpus, articles_with_name, docs_preprocessed_metrics):
        self.original_corpus = deepcopy(corpus)
        self.articles_with_name = articles_with_name
        self.articles = corpus
        self.docs_preprocessed_metrics = docs_preprocessed_metrics
        self.index = make_inverted_index(self.articles)
        self.weights = RSV_weights(self.articles, self.index)
        self.ranked = []
        self.query_text = ''
        self.N_retrieved = 0
    
        
    def RSV_doc_query(self, doc_id, query):
        '''
        This function computes the Retrieval Status Value for a given couple document - query
        using the precomputed weights
        '''
        score = 0
        doc = self.articles[doc_id]
        for term in doc:
            if term in query:
                score += self.weights[term]     
        return score

    
        
    def ranking(self, query):
        '''
        Auxiliary function for the function answer_query. Computes the score only for documents
        that are in the posting list of al least one term in the query
        '''

        docs = []
        for term in self.index: 
            if term in query:
                docs = posting_lists_union(docs, self.index[term])

        scores = []
        for doc in docs:
            scores.append((doc, self.RSV_doc_query(doc, query)))
        
        self.ranked = sorted(scores, key=lambda x: x[1], reverse = True)
        return self.ranked
    
    
    
    def recompute_weights(self, relevant_idx, query):
        '''
        Auxiliary function for relevance_feedback function and
        for the pseudo relevance feedback in answer_query function.
        Recomputes the weights, only for the terms in the query
        based on a set of relevant documents.
        '''
        
        relevant_docs = []
        for idx in relevant_idx:
          try:
            doc_id = self.ranked[idx-1][0]
            relevant_docs.append(self.articles[doc_id])
          except:
            continue
        
        N = len(self.articles)
        N_rel = len(relevant_idx)
        
        for term in query:
            if term in self.weights.keys():
                vri = 0
                for doc in relevant_docs:
                    if term in doc:
                        vri += 1
                p = (vri + 0.5) /( N_rel + 1)
                u = (DF(term, self.index) - vri + 0.5) / (N - N_rel +1)
                self.weights[term] = log((1-u)/u) + log(p/(1-p))

            
    
    def answer_query(self, query_text):
        '''
        Function to answer a free text query. Shows the first 30 words of the
        15 most relevant documents. 
        Also implements the pseudo relevance feedback with k = 5
        '''
        
        self.query_text = query_text
        query =  query_text.lower().split()
        ranking = self.ranking(query)

        ## pseudo relevance feedback 
        i = 0
        new_ranking=[]
        while i<10 and ranking != new_ranking:
            self.recompute_weights([1,2,3,4,5], query)
            new_ranking = self.ranking(query)
            i+=1


        ranking = new_ranking
        
        self.N_retrieved = 20
        
        ## print retrieved documents
        ret_list = []
        for i in range(0, min(len(ranking), self.N_retrieved)):
            article = self.original_corpus[ranking[i][0]]
            text = " ".join(article)
            #print(f"Article {self.articles_with_name[self.ranked[i][0]]}, score: {ranking[i][1]}")
            for myDoc in self.docs_preprocessed_metrics:
              if(self.articles_with_name[self.ranked[i][0]]==myDoc['Name']):
                ret_list.append((self.articles_with_name[self.ranked[i][0]], ranking[i][1], myDoc['Data']['Date'], myDoc['Data']['Location'].lower(), myDoc['Data']['Link']))
            #ret_list.append((self.articles_with_name[self.ranked[i][0]], ranking[i][1]))
            # print(text, '\n')

        self.weights = RSV_weights(self.articles, self.index)
        
        # priortitize location
        location_relevant = []
        location_irrelevant = []
        for x in ret_list:
          if u_location.lower() in x[3]:
            location_relevant.append(x)
          else:
            location_irrelevant.append(x)
        ret_list = location_relevant + location_irrelevant
        return ret_list


            
    def relevance_feedback(self, *args):
        '''
        Function that implements relevance feedback for the last query formulated.
        The set of relevant documents is given by the user through a set of indexes
        '''
        if(self.query_text == ''):
            print('Cannot get feedback before a query is formulated.')
            return
        
        relevant_idx = list(args)
        
        if(isinstance(relevant_idx[0], list)):
            relevant_idx = relevant_idx[0]
        
        query = self.query_text.upper().split()
        self.recompute_weights(relevant_idx,query)
        
        self.answer_query(self.query_text)
    
    
    def read_document(self,rank_num):
        '''
        Function that allows the user to select a document among the ones returned 
        by answer_query and read the whole text
        '''
        if(self.query_text == ''):
            print('Cannot select a document before a query is formulated.')
            return
            
        article = self.original_corpus[self.ranked[rank_num - 1][0]]
        text = " ".join(article)
        print(f"Article {rank_num}, score: {self.ranked[rank_num][1]}")
        print(text, '\n')
        
        
    def show_more(self):
        '''
        Function that allows the user to see more 10 retrieved documents
        '''
        
        if(self.N_retrieved + 10 > len(self.ranked)):
            print('No more documents available')
            return 
        
        for i in range(self.N_retrieved, self.N_retrieved+10):
            article = self.original_corpus[self.ranked[i][0]]
            if (len(article) > 30):
                article = article[0:30]
            text = " ".join(article)
            print(f"Article {self.articles_with_name[self.ranked[i][0]]}, score: {self.ranked[i][1]}")
            print(text, '\n')
        
        self.N_retrieved += 10 
        



# Example of usage

# articles, articles_with_name = import_dataset()
# bim  = BIM(articles, articles_with_name)
# bim.answer_query('Italy and Great Britain fight the enemy')
# bim.relevance_feedback(5,6,8)
# bim.show_more()
# bim.read_document(2)


# In[9]:


tweet_query = []
format = '%Y-%m-%d'
u_present_date = datetime.datetime.strptime(u_time, format)
u_prev_date = u_present_date - datetime.timedelta(days=1)
u_next_date = u_present_date + datetime.timedelta(days=1)
df_query = df.loc[df['hashtags'].str.contains(u_base_hashtag) & df['Date_Only'].isin([str(u_present_date.date()), str(u_prev_date.date()), str(u_next_date.date())])]
print(df_query.shape[0])
if df_query.shape[0]<50:
  df_query = df.loc[df['Date_Only'].isin([str(u_present_date.date()), str(u_prev_date.date()), str(u_next_date.date())])]
  df_query = df.iloc[:min(df_query.shape[0],1000),:]


# In[10]:


for tweet in df_query['Preprocessed_Data']:
  tweet_query.extend(tweet)


# In[11]:


def get_keyword_keywordextractor(df_query):
  keyword_dataset = df_query['tweet'].tolist()
  tweet_query_keyword_extractor = keyword_extractor(keyword_dataset)
  return tweet_query_keyword_extractor


# In[12]:


def get_yake_keywords(tweet_query):
  tweet_keywords_yake = []
  kw_extractor = yake.KeywordExtractor(top=20, stopwords=None)
  keywords = kw_extractor.extract_keywords(' '.join(tweet_query))
  #keywords = kw_extractor.extract_keywords(' '.join(df_query['tweet'].tolist()))
  for kw, v in keywords:
    #print("Keyphrase: ",kw, ": score", v)
    for key in kw.split():
      if(key.lower() not in tweet_keywords_yake):
        tweet_keywords_yake.append(key.lower())
  return tweet_keywords_yake


# In[13]:


def get_rake_keywords(tweet_query):
  tweet_keywords_rake = []
  rake = Rake()
  rake_keywords = rake.apply(' '.join(tweet_query).encode('ascii', 'ignore').decode())
  for kw,score in rake_keywords[:20]:
    for key in kw.split():
      if(key.lower() not in tweet_keywords_rake):
        tweet_keywords_rake.append(key.lower())
  return tweet_keywords_rake


# In[14]:


def get_text_rank_keywords(tweet_query):
  tweet_keywords_text_rank = []
  TR_keywords = summa_keywords.keywords(' '.join(tweet_query), scores=True)
  for kw,score in TR_keywords[:20]:
    for key in kw.split():
      if(key.lower() not in tweet_keywords_text_rank):
        tweet_keywords_text_rank.append(key.lower())
  return tweet_keywords_text_rank


# In[15]:


def get_keybert_keywords(tweet_query):
  # uncomment later
  keybert_model = KeyBERT(model='all-mpnet-base-v2')
  keybert_keywords = keybert_model.extract_keywords(' '.join(tweet_query), keyphrase_ngram_range=(1,1), stop_words='english', highlight=False, top_n=20)
  tweet_keywords_keybert = list(dict(keybert_keywords).keys())
  # print(tweet_keywords_keybert)
  return tweet_keywords_keybert


# In[16]:


# Example of usage
articles, articles_with_name, docs_preprocessed_metrics = import_dataset()


# # Helpful Functions

# In[17]:


def get_relevant_docs_list_for_base_hashtag(base_hashtag, base_date, docs_preprocessed_metrics):
  relevant_docs_list = []
  for doc in docs_preprocessed_metrics:
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


def nDCG(base_hashtag, base_date, prediction_list, docs_preprocessed_metrics):
  ground_truth = get_relevant_docs_list_for_base_hashtag(base_hashtag, base_date, docs_preprocessed_metrics)
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


def precision_at_k(k, base_hashtag, base_date, prediction_list, docs_preprocessed_metrics):
  relevant_docs_list = get_relevant_docs_list_for_base_hashtag(base_hashtag, base_date, docs_preprocessed_metrics)
  num_of_relevant_results=0
  list_of_rel_doc_names = []
  for x in relevant_docs_list:
    list_of_rel_doc_names.append(x[0])
  for itr in range(min(len(prediction_list), k)):
    if (prediction_list[itr][0] in list_of_rel_doc_names):
      num_of_relevant_results+=1
  return num_of_relevant_results/k


# In[20]:


def mean_average_precision(max_k, base_hashtag, base_date, relevant_docs, docs_preprocessed_metrics):
  average_precision=0
  ctr=0
  relevant_docs_list = get_relevant_docs_list_for_base_hashtag(base_hashtag, base_date, docs_preprocessed_metrics)
  for k_val in range(1,len(relevant_docs_list)+1):
    ctr+=1
    if k_val>len(relevant_docs_list):
      break
    precision_at_k_val = precision_at_k(k_val, base_hashtag, base_date, relevant_docs, docs_preprocessed_metrics)
    #print('Hashtag: {}   Precision@{}: {}'.format(base_hashtag, k_val, precision_at_k_val))
    average_precision += precision_at_k_val
  return average_precision/ctr


# In[21]:


def recall_at_k(k, base_hashtag, base_date, prediction_list, docs_preprocessed_metrics):
  relevant_docs_list = get_relevant_docs_list_for_base_hashtag(base_hashtag, base_date, docs_preprocessed_metrics)
  current_num_of_relevant_results=0
  list_of_rel_doc_names = []
  for x in relevant_docs_list:
    list_of_rel_doc_names.append(x[0])
  for itr in range(min(len(prediction_list),k)):
    if (prediction_list[itr][0] in list_of_rel_doc_names):
      current_num_of_relevant_results+=1
  if(len(relevant_docs_list)==0):
    return 0
  return current_num_of_relevant_results/len(relevant_docs_list)


# In[22]:


def mean_average_recall(max_k, base_hashtag, base_date, relevant_docs, docs_preprocessed_metrics):
  average_recall=0
  ctr=0
  relevant_docs_list = get_relevant_docs_list_for_base_hashtag(base_hashtag, base_date, docs_preprocessed_metrics)
  print(len(relevant_docs_list))
  for k_val in range(1,len(relevant_docs_list)+1):
    ctr+=1
    if k_val>len(relevant_docs_list):
      break
    recall_at_k_val = recall_at_k(k_val, base_hashtag, base_date, relevant_docs, docs_preprocessed_metrics)
    #print('Hashtag: {}   Recall@{}: {}'.format(base_hashtag, k_val, recall_at_k_val))
    average_recall += recall_at_k_val
  return average_recall/ctr


# # Plain Model (Binary Independence Model)

# In[23]:


print(articles)


# In[24]:


#results with normal query
bim  = BIM(articles, articles_with_name, docs_preprocessed_metrics)
relevant_docs_bim_plain = bim.answer_query(" ".join(tweet_query))

# bim.relevance_feedback(5,6,8)
# bim.show_more()
# bim.read_document(2)

for rank, doc in enumerate(relevant_docs_bim_plain):
  print('Rank: {} Relevant Document: {}'.format(rank+1,doc))

print()

mean_average_precision_hashtag_bim_plain = mean_average_precision(20, u_base_hashtag, u_time, relevant_docs_bim_plain, docs_preprocessed_metrics)
print('Mean Average Precision Plain Model (BIM) : {}'.format(mean_average_precision_hashtag_bim_plain))

mean_average_recall_hashtag_bim_plain = mean_average_recall(20, u_base_hashtag, u_time, relevant_docs_bim_plain, docs_preprocessed_metrics)
print('Mean Average Recall Plain Model (BIM) : {}'.format(mean_average_recall_hashtag_bim_plain))

nDCG_hashtag_bim_plain = nDCG(u_base_hashtag, u_time, relevant_docs_bim_plain, docs_preprocessed_metrics)
print('nDCG Plain Model (Binary Independence Model) : {}'.format(nDCG_hashtag_bim_plain))


# # Model with Keyword Extractor (Binary Independence Model)

# In[25]:


# results with top k keywords
bim  = BIM(articles, articles_with_name, docs_preprocessed_metrics)
tweet_query_keyword_extractor = get_keyword_keywordextractor(df_query)
relevant_docs_bim_keyword_extractor = bim.answer_query(" ".join(tweet_query_keyword_extractor[:20]))

# bim.relevance_feedback(5,6,8)
# bim.show_more()
# bim.read_document(2)

for rank, doc in enumerate(relevant_docs_bim_keyword_extractor):
  print('Rank: {} Relevant Document: {}'.format(rank+1,doc))

print()

mean_average_precision_hashtag_bim_keyword_extractor = mean_average_precision(20, u_base_hashtag, u_time, relevant_docs_bim_keyword_extractor, docs_preprocessed_metrics)
print('Mean Average Precision Keyword Extractor Model (BIM) : {}'.format(mean_average_precision_hashtag_bim_keyword_extractor))

mean_average_recall_hashtag_bim_keyword_extractor = mean_average_recall(20, u_base_hashtag, u_time, relevant_docs_bim_keyword_extractor, docs_preprocessed_metrics)
print('Mean Average Recall Keyword Extractor Model (BIM) : {}'.format(mean_average_recall_hashtag_bim_keyword_extractor))

nDCG_hashtag_bim_keyword_extractor = nDCG(u_base_hashtag, u_time, relevant_docs_bim_keyword_extractor, docs_preprocessed_metrics)
print('nDCG Keyword Extractor Model (Binary Independence Model) : {}'.format(nDCG_hashtag_bim_keyword_extractor))


# # Model with YAKE (Binary Independence Model)

# In[26]:


# results with top k keywords
bim  = BIM(articles, articles_with_name, docs_preprocessed_metrics)
tweet_keywords_yake = get_yake_keywords(tweet_query)
relevant_docs_bim_yake = bim.answer_query(" ".join(tweet_keywords_yake))

# bim.relevance_feedback(5,6,8)
# bim.show_more()
# bim.read_document(2)

for rank, doc in enumerate(relevant_docs_bim_yake):
  print('Rank: {} Relevant Document: {}'.format(rank+1,doc))

print()

mean_average_precision_hashtag_bim_yake = mean_average_precision(20, u_base_hashtag, u_time, relevant_docs_bim_yake, docs_preprocessed_metrics)
print('Mean Average Precision YAKE Model (BIM) : {}'.format(mean_average_precision_hashtag_bim_yake))

mean_average_recall_hashtag_bim_yake = mean_average_recall(20, u_base_hashtag, u_time, relevant_docs_bim_yake, docs_preprocessed_metrics)
print('Mean Average Recall YAKE Model (BIM) : {}'.format(mean_average_recall_hashtag_bim_yake))

nDCG_hashtag_bim_yake = nDCG(u_base_hashtag, u_time, relevant_docs_bim_yake, docs_preprocessed_metrics)
print('nDCG YAKE Model (Binary Independence Model) : {}'.format(nDCG_hashtag_bim_yake))


# # Model with RAKE (Binary Independence Model)

# In[27]:


# results with top k keywords
bim  = BIM(articles, articles_with_name, docs_preprocessed_metrics)
tweet_keywords_rake = get_rake_keywords(tweet_query)
relevant_docs_bim_rake = bim.answer_query(" ".join(tweet_keywords_rake))

# bim.relevance_feedback(5,6,8)
# bim.show_more()
# bim.read_document(2)

for rank, doc in enumerate(relevant_docs_bim_rake):
  print('Rank: {} Relevant Document: {}'.format(rank+1,doc))

print()

mean_average_precision_hashtag_bim_rake = mean_average_precision(20, u_base_hashtag, u_time, relevant_docs_bim_rake, docs_preprocessed_metrics)
print('Mean Average Precision RAKE Model (BIM) : {}'.format(mean_average_precision_hashtag_bim_rake))

mean_average_recall_hashtag_bim_rake = mean_average_recall(20, u_base_hashtag, u_time, relevant_docs_bim_rake, docs_preprocessed_metrics)
print('Mean Average Recall RAKE Model (BIM) : {}'.format(mean_average_recall_hashtag_bim_rake))

nDCG_hashtag_bim_rake = nDCG(u_base_hashtag, u_time, relevant_docs_bim_rake, docs_preprocessed_metrics)
print('nDCG RAKE Model (Binary Independence Model) : {}'.format(nDCG_hashtag_bim_rake))


# # Model with TextRank

# In[28]:


# results with top k keywords
bim  = BIM(articles, articles_with_name, docs_preprocessed_metrics)
tweet_keywords_text_rank = get_yake_keywords(tweet_query)
relevant_docs_bim_text_rank = bim.answer_query(" ".join(tweet_keywords_text_rank))

# bim.relevance_feedback(5,6,8)
# bim.show_more()
# bim.read_document(2)

for rank, doc in enumerate(relevant_docs_bim_text_rank):
  print('Rank: {} Relevant Document: {}'.format(rank+1,doc))

print()

mean_average_precision_hashtag_bim_text_rank = mean_average_precision(20, u_base_hashtag, u_time, relevant_docs_bim_text_rank, docs_preprocessed_metrics)
print('Mean Average Precision TextRank Model (BIM) : {}'.format(mean_average_precision_hashtag_bim_text_rank))

mean_average_recall_hashtag_bim_text_rank = mean_average_recall(20, u_base_hashtag, u_time, relevant_docs_bim_text_rank, docs_preprocessed_metrics)
print('Mean Average Recall TextRank Model (BIM) : {}'.format(mean_average_recall_hashtag_bim_text_rank))

nDCG_hashtag_bim_text_rank = nDCG(u_base_hashtag, u_time, relevant_docs_bim_text_rank, docs_preprocessed_metrics)
print('nDCG TextRank Model (Binary Independence Model) : {}'.format(nDCG_hashtag_bim_text_rank))


# # Model with KeyBert

# In[29]:


# results with top k keywords
bim  = BIM(articles, articles_with_name, docs_preprocessed_metrics)
tweet_keywords_keybert = get_keybert_keywords(tweet_query)
relevant_docs_bim_keybert = bim.answer_query(" ".join(tweet_keywords_keybert))

# bim.relevance_feedback(5,6,8)
# bim.show_more()
# bim.read_document(2)

for rank, doc in enumerate(relevant_docs_bim_keybert):
  print('Rank: {} Relevant Document: {}'.format(rank+1,doc))

print()

mean_average_precision_hashtag_bim_keybert = mean_average_precision(20, u_base_hashtag, u_time, relevant_docs_bim_keybert, docs_preprocessed_metrics)
print('Mean Average Precision KeyBERT Model (BIM) : {}'.format(mean_average_precision_hashtag_bim_keybert))

mean_average_recall_hashtag_bim_keybert = mean_average_recall(20, u_base_hashtag, u_time, relevant_docs_bim_keybert, docs_preprocessed_metrics)
print('Mean Average Recall KeyBERT Model (BIM) : {}'.format(mean_average_recall_hashtag_bim_keybert))

nDCG_hashtag_bim_keybert = nDCG(u_base_hashtag, u_time, relevant_docs_bim_keybert, docs_preprocessed_metrics)
print('nDCG KeyBERT Model (Binary Independence Model) : {}'.format(nDCG_hashtag_bim_keybert))


# # Across all hashtags

# In[30]:


global_list =[['IPL', '2022-02-25', 'india'], ['IPL', '2022-02-16', 'india'], ['Drishyam2', '2022-02-17', 'india'], ['elections', '2022-02-27', 'imphal'], ['elections', '2022-02-27', 'new delhi'], ['elections', '2022-02-27', 'uttar pradesh'], ['nursesstrike', '2022-02-15', 'australia'], ['Djokovic', '2022-02-15', 'australia'],['hijab', '2022-02-19', 'india'], ['msdtrong', '2022-02-14', 'Florida'], ['ottawaoccupied', '2022-02-14', 'Ottawa'], ['Boycott_ChennaiSuperKings', '2022-02-14', 'chennai'], ['GlanceJio', '2022-02-14', 'New Delhi'], ['ArabicKuthu', '2022-02-14', 'CHENNAI'], ['Djokovic', '2022-02-15', 'AUSTRALIA'], ['Real Madrid', '2022-02-15', 'Madrid'], ['bighit', '2022-02-15', 'ASSAM'], ['Maxwell', '2022-02-15', 'india'], ['mafsau', '2022-02-16', 'australia'], ['channi', '2022-02-16', 'punjab'], ['ayalaan', '2022-02-16', 'tamil nadu'], ['jkbose', '2022-02-16', 'srinagar'], ['happybirthdayjhope', '2022-02-17', 'new delhi'], ['mohsinbaig', '2022-02-17', 'islamabad'],  ['ShivajiJayanti', '2022-02-18', 'maharashtra'], ['OperationDudula', '2022-02-20', 'South Africa'], ['UFCVegas48', '2022-02-20', 'india'], ['FCNPSG', '2022-02-20', 'PARIS'], ['shivamogga', '2022-02-21', 'KARNATAKA'],  ['StayAlive_CHAKHO', '2022-02-22', 'seoul'], ['KaranSinghGrover', '2022-02-22', 'india'], ['NationalMargaritaDay', '2022-02-22', 'United States of America'], ['dontsaygay', '2022-02-22', 'New York'], ['RIPRikyRick', '2022-02-23', 'South Africa'], ['budget2022', '2022-02-23', 'Cape Town'], ['CottonFest2022', '2022-02-23', 'nigeria'], ['NationalChiliDay', '2022-02-24', 'Ohio'], ['stockmarketcrash', '2022-02-24', 'mumbai'], ['BidenisaFailure', '2022-02-24', 'United States of America']]
print(len(global_list))
global_list = global_list[:30]


# In[42]:


global_average_mean_average_precision_plain = []
global_mean_average_recall_plain = []
global_mean_average_ndcg_plain = []

global_average_mean_average_precision_keyword_extractor = []
global_mean_average_recall_keyword_extractor = []
global_mean_average_ndcg_keyword_exractor = []

global_average_mean_average_precision_rake = []
global_mean_average_recall_rake = []
global_mean_average_ndcg_rake = []

global_average_mean_average_precision_yake = []
global_mean_average_recall_yake = []
global_mean_average_ndcg_yake = []

global_average_mean_average_precision_text_rank = []
global_mean_average_recall_text_rank = []
global_mean_average_ndcg_text_rank = []


global_average_mean_average_precision_keybert = []
global_mean_average_recall_keybert = []
global_mean_average_ndcg_keybert = []

for iter in tqdm(range(len(global_list))):
  u_base_hashtag = global_list[iter][0]
  u_time = global_list[iter][1]
  u_location = global_list[iter][2]
  tweet_query = []
  format = '%Y-%m-%d'
  u_present_date = datetime.datetime.strptime(u_time, format)
  u_prev_date = u_present_date - datetime.timedelta(days=1)
  u_next_date = u_present_date + datetime.timedelta(days=1)
  df_query = df.loc[df['hashtags'].str.contains(u_base_hashtag) & df['Date_Only'].isin([str(u_present_date.date()), str(u_prev_date.date()), str(u_next_date.date())])]

  for tweet in df_query['Preprocessed_Data']:
    tweet_query.extend(tweet)
  
  tweet_keywords = []
  kw_extractor = yake.KeywordExtractor(top=20, stopwords=None)
  keywords = kw_extractor.extract_keywords(' '.join(tweet_query))
  for kw, v in keywords:
    #print("Keyphrase: ",kw, ": score", v)
    for key in kw.split():
      if(key not in tweet_keywords):
        tweet_keywords.append(key)
  
  docs_preprocessed = []

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
  
  # Plain
  articles, articles_with_name, docs_preprocessed_metrics = import_dataset()
  bim  = BIM(articles, articles_with_name,docs_preprocessed_metrics)
  relevant_articles_list_plain = bim.answer_query(" ".join(tweet_query))
  mean_average_precision_hashtag_plain = mean_average_precision(20, u_base_hashtag, u_time, relevant_articles_list_plain, docs_preprocessed_metrics)
  global_average_mean_average_precision_plain.append(mean_average_precision_hashtag_plain)
  mean_average_recall_hashtag_plain = mean_average_recall(20, u_base_hashtag, u_time, relevant_articles_list_plain, docs_preprocessed_metrics)
  global_mean_average_recall_plain.append(mean_average_recall_hashtag_plain)
  nDCG_hashtag_plain = nDCG(u_base_hashtag, u_time, relevant_articles_list_plain, docs_preprocessed)
  global_mean_average_ndcg_plain.append(nDCG_hashtag_plain)

  # keyword extractor
  bim  = BIM(articles, articles_with_name, docs_preprocessed_metrics)
  tweet_query_keyword_extractor = get_keyword_keywordextractor(df_query)
  relevant_articles_list_keyword_extractor = bim.answer_query(" ".join(tweet_query_keyword_extractor[:20]))
  mean_average_precision_hashtag_keyword_extractor = mean_average_precision(20, u_base_hashtag, u_time, relevant_articles_list_keyword_extractor, docs_preprocessed_metrics)
  global_average_mean_average_precision_keyword_extractor.append(mean_average_precision_hashtag_keyword_extractor)
  mean_average_recall_hashtag_keyword_extractor = mean_average_recall(20, u_base_hashtag, u_time, relevant_articles_list_keyword_extractor, docs_preprocessed_metrics)
  global_mean_average_recall_keyword_extractor.append(mean_average_recall_hashtag_keyword_extractor)
  nDCG_hashtag_keyword_extractor = nDCG(u_base_hashtag, u_time, relevant_articles_list_keyword_extractor, docs_preprocessed)
  global_mean_average_ndcg_keyword_exractor.append(nDCG_hashtag_keyword_extractor)

  # Rake
  bim  = BIM(articles, articles_with_name, docs_preprocessed_metrics)
  tweet_query_rake = get_rake_keywords(tweet_query)
  relevant_articles_list_rake = bim.answer_query(" ".join(tweet_query_rake))
  mean_average_precision_hashtag_rake = mean_average_precision(20, u_base_hashtag, u_time, relevant_articles_list_rake, docs_preprocessed_metrics)
  global_average_mean_average_precision_rake.append(mean_average_precision_hashtag_rake)
  mean_average_recall_hashtag_rake = mean_average_recall(20, u_base_hashtag, u_time, relevant_articles_list_rake, docs_preprocessed_metrics)
  global_mean_average_recall_rake.append(mean_average_recall_hashtag_rake)
  nDCG_hashtag_rake = nDCG(u_base_hashtag, u_time,  relevant_articles_list_rake, docs_preprocessed)
  global_mean_average_ndcg_rake.append(nDCG_hashtag_rake)

  # Yake
  bim  = BIM(articles, articles_with_name, docs_preprocessed_metrics)
  tweet_query_yake = get_yake_keywords(tweet_query)
  relevant_articles_list_yake = bim.answer_query(" ".join(tweet_query_yake))
  mean_average_precision_hashtag_yake = mean_average_precision(20, u_base_hashtag, u_time, relevant_articles_list_yake, docs_preprocessed_metrics)
  global_average_mean_average_precision_yake.append(mean_average_precision_hashtag_yake)
  mean_average_recall_hashtag_yake = mean_average_recall(20, u_base_hashtag, u_time, relevant_articles_list_yake, docs_preprocessed_metrics)
  global_mean_average_recall_yake.append(mean_average_recall_hashtag_yake)
  nDCG_hashtag_yake = nDCG(u_base_hashtag, u_time, relevant_articles_list_yake, docs_preprocessed)
  global_mean_average_ndcg_yake.append(nDCG_hashtag_yake)

  # Text Rank
  bim  = BIM(articles, articles_with_name, docs_preprocessed_metrics)
  tweet_query_text_rank = get_text_rank_keywords(tweet_query)
  relevant_articles_list_text_rank = bim.answer_query(" ".join(tweet_query_text_rank))
  mean_average_precision_hashtag_text_rank = mean_average_precision(20, u_base_hashtag, u_time, relevant_articles_list_text_rank, docs_preprocessed_metrics)
  global_average_mean_average_precision_text_rank.append(mean_average_precision_hashtag_text_rank)
  mean_average_recall_hashtag_text_rank = mean_average_recall(20, u_base_hashtag, u_time, relevant_articles_list_text_rank, docs_preprocessed_metrics)
  global_mean_average_recall_text_rank.append(mean_average_recall_hashtag_text_rank)
  nDCG_hashtag_text_rank = nDCG(u_base_hashtag, u_time, relevant_articles_list_text_rank, docs_preprocessed)
  global_mean_average_ndcg_text_rank.append(nDCG_hashtag_text_rank)
  
  
  # Keybert
  bim  = BIM(articles, articles_with_name, docs_preprocessed_metrics)
  tweet_query_keybert = get_keybert_keywords(tweet_query)
  relevant_articles_list_keybert = bim.answer_query(" ".join(tweet_query_keybert))
  mean_average_precision_hashtag_keybert = mean_average_precision(20, u_base_hashtag, u_time, relevant_articles_list_keybert, docs_preprocessed_metrics)
  global_average_mean_average_precision_keybert.append(mean_average_precision_hashtag_keybert)
  mean_average_recall_hashtag_keybert = mean_average_recall(20, u_base_hashtag, u_time, relevant_articles_list_keybert, docs_preprocessed_metrics)
  global_mean_average_recall_keybert.append(mean_average_recall_hashtag_keybert)
  nDCG_hashtag_keybert = nDCG(u_base_hashtag, u_time, relevant_articles_list_keybert, docs_preprocessed)
  global_mean_average_ndcg_keybert.append(nDCG_hashtag_keybert)


# In[36]:


overall_average_mean_average_precision_plain = sum(global_average_mean_average_precision_plain)/len(global_average_mean_average_precision_plain)
print(overall_average_mean_average_precision_plain)

overall_mean_average_recall_plain = sum(global_mean_average_recall_plain)/len(global_mean_average_recall_plain)
print(overall_mean_average_recall_plain)

overall_mean_average_ndcg_plain = sum(global_mean_average_ndcg_plain)/len(global_mean_average_ndcg_plain)
print(overall_mean_average_ndcg_plain)


# In[37]:


overall_average_mean_average_precision_keyword_extractor = sum(global_average_mean_average_precision_keyword_extractor)/len(global_average_mean_average_precision_keyword_extractor)
print(overall_average_mean_average_precision_keyword_extractor)

overall_mean_average_recall_keyword_extractor = sum(global_mean_average_recall_keyword_extractor)/len(global_mean_average_recall_keyword_extractor)
print(overall_mean_average_recall_keyword_extractor)

overall_mean_average_ndcg_keyword_extractor = sum(global_mean_average_ndcg_keyword_exractor)/len(global_mean_average_ndcg_keyword_exractor)
print(overall_mean_average_ndcg_keyword_extractor)


# In[38]:


overall_average_mean_average_precision_rake = sum(global_average_mean_average_precision_rake)/len(global_average_mean_average_precision_rake)
print(overall_average_mean_average_precision_rake)

overall_mean_average_recall_rake = sum(global_mean_average_recall_rake)/len(global_mean_average_recall_rake)
print(overall_mean_average_recall_rake)

overall_mean_average_ndcg_rake = sum(global_mean_average_ndcg_rake)/len(global_mean_average_ndcg_rake)
print(overall_mean_average_ndcg_rake)


# In[39]:


overall_average_mean_average_precision_yake = sum(global_average_mean_average_precision_yake)/len(global_average_mean_average_precision_yake)
print(overall_average_mean_average_precision_yake)

overall_mean_average_recall_yake = sum(global_mean_average_recall_yake)/len(global_mean_average_recall_yake)
print(overall_mean_average_recall_yake)

overall_mean_average_ndcg_yake = sum(global_mean_average_ndcg_yake)/len(global_mean_average_ndcg_yake)
print(overall_mean_average_ndcg_yake)


# In[40]:


overall_average_mean_average_precision_text_rank = sum(global_average_mean_average_precision_text_rank)/len(global_average_mean_average_precision_text_rank)
print(overall_average_mean_average_precision_text_rank)

overall_mean_average_recall_text_rank = sum(global_mean_average_recall_text_rank)/len(global_mean_average_recall_text_rank)
print(overall_mean_average_recall_text_rank)


overall_mean_average_ndcg_text_rank = sum(global_mean_average_ndcg_text_rank)/len(global_mean_average_ndcg_text_rank)
print(overall_mean_average_ndcg_text_rank)


# In[41]:


overall_average_mean_average_precision_keybert = sum(global_average_mean_average_precision_keybert)/len(global_average_mean_average_precision_keybert)
print(overall_average_mean_average_precision_keybert)

overall_mean_average_recall_keybert = sum(global_mean_average_recall_keybert)/len(global_mean_average_recall_keybert)
print(overall_mean_average_recall_keybert)

overall_mean_average_ndcg_keybert = sum(global_mean_average_ndcg_keybert)/len(global_mean_average_ndcg_keybert)
print(overall_mean_average_ndcg_keybert)

