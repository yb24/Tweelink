from fastapi import FastAPI,Request
import datetime
import pickle
import json
from glob import glob
import os
import yake
import pandas as pd
import numpy as np
import csv
import json
from itertools import islice
from collections import OrderedDict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
import datetime
from fastapi.middleware.cors import CORSMiddleware



#Importing models
import final_results_jaccard_coefficient as jaccard
import final_results_cosine_similarity_count_vectorizer as cosine_count
import final_results_cosine_similarity_tfidf_vectorizer as cosine_tf
import final_results_binary_independence as binary_independence
import final_results_okapi_bm25 as bm25
import final_results_tf_idf_binary as tf1
import final_results_tf_idf_double_normalization as tf2
import final_results_tf_idf_log_normalization as tf3
import final_results_tf_idf_raw_count as tf4
import final_results_tf_idf_term_frequency as tf5


app = FastAPI()


origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



#Loading twitter dataset
file1 = open("/Users/harmansingh/Desktop/IR/Tweelink/Web/db/twitter_base_preprocessed.pkl", "rb")
df = pickle.load(file1)
file1.close()




format = '%Y-%m-%d'


hashtags = ['SuisseSecrets', 'MCITOT', 'KPACLalitha', 'TeamGirlsCup', 'wordle251', 'aewdynamite', 'NFU22', 'RehmanMalik', 'thewifeshowmax', 'IndiaFightsCorona', 'nadal', 'GoPackGo', 'ElonMusk', 'stormfranklin', 'Wordle242', 'CarabaoCupFinal', 'downingstreetbriefing', 'acemagashule', 'AmaalMallik', 'narendramodi', 'balakotairstrike', 'zeynep', 'happybirthdayjhope', 'StayAlive_CHAKHO', 'BattleOfBillingsBridge', 'BorisJohnsonResign', 'Adipurush', 'GOPtheRussianParty', 'SaluteToBrave', 'PushpaTheRule', 'stopwar', 'scottytheannouncer', 'Drishyam2', 'freedomconvoy22', '2002GujaratGenocide', 'UkraineRussiaCrisis', 'JusticeForKhojaly', 'sandhyamukherjee', 'TaylorSwift', 'InternalAssessmentForAll2022', 'nursesstrike', 'kfcindia', 'vaccine', 'IPL', 'MWC22', 'BRENEW', 'EVEMCI', 'OscarsFanFavorite', 'dontsaygay', 'JusticeForDinesh', 'OPSRajasthan', 'RandomActsOfKindnessDay', 'ahmedabadblast2008', 'BBMzansi', 'PuneethRajkumar', 'INDvWI', 'StormEunice', 'mafsau', 'F1Testing', 'harisrauf', 'SurpriseDay', 'masterchefsa', 'BachchhanPaandey', 'MGP2022', 'MSC2022', 'SCOvsFRA', 'superrugbypacific', 'munwat', 'ArabicKuthu', 'CaringForDerek', 'HorizonForbiddenWest', 'JohnsonOut21', 'ARSWOL', 'RareDiseaseDay', 'Realme9ProPlus', 'mohsinbaig', 'HappyBirthdayAlyGoni', 'SecretsOfDumbledore', 'NeverForget', 'PSLFinal', 'SUHOs_Resume_For_EXOL', 'Canucks', 'FCNPSG', 'PMIKaddressToNation', 'IPLAuction', 'PopularFrontDay', 'VeerSavarkar', 'NapoliBarca', 'IPCC', 'BoycottWalgreens', 'ShivajiJayanti', 'skeemsaam', 'justiceforharsha', 'Euphoria', 'ottawaoccupied', 'CottonFest2022', 'MultiverseOfMadness', 'RIPRikyRick', 'DrDre', 'stockmarketcrash', 'tommytiernanshow', 'AK61', 'CallTheMidwife', 'msdstrong', 'CanadaHasFallen', 'lahoreqalandars', 'AhmaudArbery', 'jkbose', 'AhmedabadFoundationDay', 'OHN22', 'ScottyFromWelding', 'truthsocial', 'HappyBirthdayABD', 'IREvITA', 'VERZUZ', 'ShameOnBirenSingh', 'sundowns', 'RussiaUkraineCrisis', 'ayalaan', 'Cyberpunk2077', 'PunjabElections2022', 'hijab', 'kejriwalvsall', 'PunjabElections', 'TheBatman', 'bighit', 'Eminem', 'elections', 'jeremyvine', 'Maxwell', 'GlanceJio', 'UntamedKaziranga', 'CBI5TheBrain', 'WriddhimanSaha', 'PlotToKillModi', 'PunjabPanjeNaal', 'NationalScienceDay', '25YearsOfYuvanism', 'UFCVegas48', 'SCOTUS', 'channi', 'BlackDay', 'EFFvsAfriforum', 'DragRaceUK', 'HemanandaBiswal', 'IshanKishan', 'MUNBHA', 'HalftimeShow', 'StormDudley', 'Dhoni', 'AAP4Khalistan', 'HBDTNCM', 'shivamogga', 'NationalMargaritaDay', 'LeafsForever', 'BidenisaFailure', 'NZvSA', 'StrangerThings', 'Djokovic', 'PolioDrops', 'Boycott_ChennaiSuperKings', 'Valimai', 'nbaallstar', 'aaraattu', 'Rosettenville', 'BalikaVadhu2OnVoot', 'VisionIAS', 'WorldBookDay', 'OttawaStrong', 'HappyBirthdayKCR', 'BJPwinningUP', 'RavidasJayanti', 'Real Madrid', 'EuropaLeague', 'FullyFaltooNFTdrop', 'ShabeMeraj', 'IStandWithTrudeau', 'MirabaiChanu', 'NationalChiliDay', 'StopPutinNOW', 'PinkShirtDay', '90dayfiance', 'KarnatakaHijabControversy', 'SuperBowl', 'budget2022', 'ProKabbadi', 'OperationDudula', 'bigbrothermzansi', 'DelightEveryMoment', 'ODDINARY', 'AmazonFabTVFest', 'PrayForPeace', 'WorldNGODay', 'UltimateDraft', 'MillionAtIndiaPavilion', 'BlandDoritos', 'PokemonDay', 'Punjab_With_Modi', 'HappyBirthdayPrinceSK', 'QueenElizabeth', 'TrudeauTyranny', 'KaranSinghGrover', 'BappiLahiri', 'KejriwalSupportsKhalistan', 'modi', 'OperationGanga', 'worldwar3', 'NationalDrinkWineDay', 'UFCVegas49', 'billgates', 'AustraliaDecides', 'DeepSidhu']

date_high = datetime.datetime(2022, 3, 1, 0, 0)
date_low = datetime.datetime(2022, 2, 13, 0, 0)



#Loading keybert_model
keybert_model = KeyBERT(model='all-mpnet-base-v2')


#SOTA Keyword extractor
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




#Getting SOTA keywords (Takes df_query)
def get_keywords_sota(df_query):
    keyword_dataset = df_query['tweet'].tolist()
    tweet_query_keyword_extractor = keyword_extractor(keyword_dataset)
    return tweet_query_keyword_extractor


#Getting YAKE keywords (Takes tweet_query)
def get_keywords_yake(tweet_query):
    tweet_keywords_yake = []
    kw_extractor = yake.KeywordExtractor(top=20, stopwords=None)
    keywords = kw_extractor.extract_keywords(' '.join(tweet_query))
    for kw, v in keywords:
        for key in kw.split():
            if(key.lower() not in tweet_keywords_yake):
                tweet_keywords_yake.append(key.lower())
    return tweet_keywords_yake


#Getting RAKE keywords (Takes tweet_query)
def get_keywords_rake(tweet_query):
    tweet_keywords_rake = []
    rake = Rake()
    rake_keywords = rake.apply(' '.join(tweet_query).encode('ascii', 'ignore').decode())
    for kw,score in rake_keywords[:20]:
        for key in kw.split():
            if(key.lower() not in tweet_keywords_rake):
                tweet_keywords_rake.append(key.lower())
    return tweet_keywords_rake


#Getting TextRank keywords (Takes tweet_query)
def get_keywords_text_rank(tweet_query):
    tweet_keywords_text_rank = []
    TR_keywords = summa_keywords.keywords(' '.join(tweet_query), scores=True)
    for kw,score in TR_keywords[:20]:
        for key in kw.split():
            if(key.lower() not in tweet_keywords_text_rank):
                tweet_keywords_text_rank.append(key.lower())
    return tweet_keywords_text_rank



#Getting TextRank keywords (Takes tweet_query)
def get_keywords_keybert(tweet_query):
    keybert_keywords = keybert_model.extract_keywords(' '.join(tweet_query), keyphrase_ngram_range=(1,1), stop_words='english', highlight=False, top_n=20)
    tweet_keywords_keybert = list(dict(keybert_keywords).keys())
    return tweet_keywords_keybert



@app.get("/")
async def root():
    return {"message": "Welcome to Tweelink"}


#Loading articles dataset
def get_docs_preprocessed(u_present_date, u_prev_date, u_next_date):
    docs_preprocessed = []
    total_documents = 0
    path = '/Users/harmansingh/Desktop/IR/Tweelink/Web/db/Tweelink_Articles_Processed'
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
    return docs_preprocessed, total_documents



#Importing articles for BIP
def import_dataset_BIP(u_present_date, u_prev_date, u_next_date):
    """
    This function import all the articles in the TIME corpus,
    returning list of lists where each sub-list contains all the
    terms present in the document as a string.
    """

    docs_preprocessed_metrics = []
    docs_preprocessed = []
    docs_preprocessed_with_names = []
    path = '/Users/harmansingh/Desktop/IR/Tweelink/Web/db/Tweelink_Articles_Processed'
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





#Jaccard
def jaccard_results(docs_preprocessed, tweet_query, u_base_hashtag, u_time, u_location):
    relevant_docs_plain = jaccard.find_relevant_documents(docs_preprocessed, tweet_query, u_base_hashtag, u_time, u_location)
    return relevant_docs_plain


#Cosine: Count vectorizer
def cosine_count_results(docs_preprocessed, tweet_query, u_base_hashtag, u_time, u_location):
    relevant_docs_plain = cosine_count.find_relevant_documents_cosine_similarity_count_vectorizer(docs_preprocessed, tweet_query, u_base_hashtag, u_time, u_location)
    return relevant_docs_plain

#Cosine: Tf-idf vectorizer
def cosine_tf_results(docs_preprocessed, tweet_query, u_base_hashtag, u_time, u_location):
    relevant_docs_plain = cosine_tf.find_relevant_documents_cosine_similarity_tfidf_vectorizer(docs_preprocessed, tweet_query, u_base_hashtag, u_time, u_location)
    return relevant_docs_plain

#Binary Independence (ERROR!)
def bip_results(tweet_query, u_base_hashtag, u_time, u_location):
    u_present_date = datetime.datetime.strptime(u_time, format)
    u_prev_date = u_present_date - datetime.timedelta(days=1)
    u_next_date = u_present_date + datetime.timedelta(days=1)
    articles, articles_with_name, docs_preprocessed_metrics = import_dataset_BIP(u_present_date, u_prev_date, u_next_date)
    bim  = binary_independence.BIM(articles, articles_with_name, docs_preprocessed_metrics)
    relevant_docs_bim_plain = bim.answer_query(" ".join(tweet_query), u_location)
    return relevant_docs_bim_plain

#BM25 
def bm25_results(docs_preprocessed, tweet_query, u_base_hashtag, u_time, u_location):
    relevant_docs_plain = bm25.find_relevant_documents_okapibm25(docs_preprocessed, tweet_query, u_base_hashtag, u_time, u_location)
    return relevant_docs_plain

#TF-IDF: Binary 
def tf_idf_1(docs_preprocessed, tweet_query, u_base_hashtag, u_time, u_location, isPlain, total_documents):

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

    max_freq_word_in_doc = []
    for words_in_doc in word_doc_matrix:
        max_freq_word_in_doc.append(np.max(words_in_doc))
    
    IDF = {}
    inverted_index_final = {}
    for key in inverted_index:
        inverted_index_final[key] = {"Frequency": len(inverted_index[key]), "doc_id":inverted_index[key]}
        IDF[key] = np.log10(total_documents/(len(inverted_index[key]) +1))

    #  sorting according to words
    inverted_index_final = dict(sorted(inverted_index_final.items()))
    Binary_tf_idf = tf1.Binary_tf_idf_scheme(np.copy(word_doc_matrix))

    for i in tqdm(range(len(vocabulary))):
        IDF_factor = IDF[id2word[i]]
        Binary_tf_idf[:,i] *= IDF_factor

    relevant_docs_plain = tf1.find_relevant_documents_tfidf_binary(isPlain, docs_preprocessed, tweet_query, u_base_hashtag, u_time, u_location, vocabulary, word2id, id2word, IDF, Binary_tf_idf)
    return relevant_docs_plain


#TF-IDF: Double Normalization
def tf_idf_2(docs_preprocessed, tweet_query, u_base_hashtag, u_time, u_location, isPlain, total_documents):

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

    max_freq_word_in_doc = []
    for words_in_doc in word_doc_matrix:
        max_freq_word_in_doc.append(np.max(words_in_doc))
    
    IDF = {}
    inverted_index_final = {}
    for key in inverted_index:
        inverted_index_final[key] = {"Frequency": len(inverted_index[key]), "doc_id":inverted_index[key]}
        IDF[key] = np.log10(total_documents/(len(inverted_index[key]) +1))

    #  sorting according to words
    inverted_index_final = dict(sorted(inverted_index_final.items()))
    Double_Normalization_tf_idf = tf2.Double_Normalization_tf_idf_scheme(np.copy(word_doc_matrix), total_documents, max_freq_word_in_doc)
    for i in tqdm(range(len(vocabulary))):
        IDF_factor = IDF[id2word[i]]
        Double_Normalization_tf_idf[:, i] *= IDF_factor

    relevant_docs_plain = tf2.find_relevant_documents_tfidf_double_normalization(isPlain, docs_preprocessed, tweet_query, u_base_hashtag, u_time, u_location, vocabulary, word2id, id2word, IDF, Double_Normalization_tf_idf)
    return relevant_docs_plain


#TF-IDF: Log Normalization
def tf_idf_3(docs_preprocessed, tweet_query, u_base_hashtag, u_time, u_location, isPlain, total_documents):

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

    max_freq_word_in_doc = []
    for words_in_doc in word_doc_matrix:
        max_freq_word_in_doc.append(np.max(words_in_doc))
    
    IDF = {}
    inverted_index_final = {}
    for key in inverted_index:
        inverted_index_final[key] = {"Frequency": len(inverted_index[key]), "doc_id":inverted_index[key]}
        IDF[key] = np.log10(total_documents/(len(inverted_index[key]) +1))

    #  sorting according to words
    inverted_index_final = dict(sorted(inverted_index_final.items()))
    Log_Normalization_tf_idf = tf3.Log_Normalization_tf_idf_scheme(np.copy(word_doc_matrix))
    for i in tqdm(range(len(vocabulary))):
        IDF_factor = IDF[id2word[i]]
        Log_Normalization_tf_idf[:, i] *= IDF_factor

    relevant_docs_plain = tf3.find_relevant_documents_tfidf_log_normalization(isPlain, docs_preprocessed, tweet_query, u_base_hashtag, u_time, u_location, vocabulary, word2id, id2word, IDF, Log_Normalization_tf_idf)
    return relevant_docs_plain


#TF-IDF: Raw Count
def tf_idf_4(docs_preprocessed, tweet_query, u_base_hashtag, u_time, u_location, isPlain, total_documents):

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

    max_freq_word_in_doc = []
    for words_in_doc in word_doc_matrix:
        max_freq_word_in_doc.append(np.max(words_in_doc))
    
    IDF = {}
    inverted_index_final = {}
    for key in inverted_index:
        inverted_index_final[key] = {"Frequency": len(inverted_index[key]), "doc_id":inverted_index[key]}
        IDF[key] = np.log10(total_documents/(len(inverted_index[key]) +1))

    #  sorting according to words
    inverted_index_final = dict(sorted(inverted_index_final.items()))
    Raw_count_tf_idf = tf4.Raw_count_tf_idf_scheme(np.copy(word_doc_matrix))
    for i in tqdm(range(len(vocabulary))):
        IDF_factor = IDF[id2word[i]]
        Raw_count_tf_idf[:,i] *= IDF_factor

    relevant_docs_plain = tf4.find_relevant_documents_tfidf_raw_count(isPlain, docs_preprocessed, tweet_query, u_base_hashtag, u_time, u_location, vocabulary, word2id, id2word, IDF, Raw_count_tf_idf)
    return relevant_docs_plain



#TF-IDF: Term Frequency
def tf_idf_5(docs_preprocessed, tweet_query, u_base_hashtag, u_time, u_location, isPlain, total_documents):

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

    max_freq_word_in_doc = []
    for words_in_doc in word_doc_matrix:
        max_freq_word_in_doc.append(np.max(words_in_doc))
    
    IDF = {}
    inverted_index_final = {}
    for key in inverted_index:
        inverted_index_final[key] = {"Frequency": len(inverted_index[key]), "doc_id":inverted_index[key]}
        IDF[key] = np.log10(total_documents/(len(inverted_index[key]) +1))

    #  sorting according to words
    inverted_index_final = dict(sorted(inverted_index_final.items()))
    Term_frequency_tf_idf = tf5.Term_frequency_tf_idf_scheme(np.copy(word_doc_matrix), total_documents, total_words_in_doc)
    for i in tqdm(range(len(vocabulary))):
        IDF_factor = IDF[id2word[i]]
        Term_frequency_tf_idf[:,i] *= IDF_factor

    relevant_docs_plain = tf5.find_relevant_documents_tfidf_term_frequency(isPlain, docs_preprocessed, tweet_query, u_base_hashtag, u_time, u_location, vocabulary, word2id, id2word, IDF,Term_frequency_tf_idf)
    return relevant_docs_plain

@app.get("/result")
async def root(req: Request):
    request_args = dict(req.query_params)

    u_base_hashtag = request_args['hashtag']
    u_time = request_args['date']
    u_location = request_args['location']
    u_model = request_args['model']
    u_keyword_extractor = request_args['keyword']


    #Checking if hashtag is present in our list or not
    if(u_base_hashtag not in hashtags):
        return ['Hashtag not found, Google Search Instead?']

    

    tweet_query = []
    format = '%Y-%m-%d'
    u_present_date = datetime.datetime.strptime(u_time, format)
    u_prev_date = u_present_date - datetime.timedelta(days=1)
    u_next_date = u_present_date + datetime.timedelta(days=1)


    #Checking if the dates are withing the range [14 Feb - 28 March]
    if(u_present_date <= date_low or u_present_date>=date_high):
        return ['Date out of range, Google Search Instead?']



    df_query = df.loc[df['hashtags'].str.contains(u_base_hashtag) & df['Date_Only'].isin([str(u_present_date.date()), str(u_prev_date.date()), str(u_next_date.date())])]
    if df_query.shape[0]<50:
        df_query = df.loc[df['Date_Only'].isin([str(u_present_date.date()), str(u_prev_date.date()), str(u_next_date.date())])]
        df_query = df.iloc[:min(df_query.shape[0],1000),:]

    for tweet in df_query['Preprocessed_Data']:
        tweet_query.extend(tweet)


    if(u_keyword_extractor=="KECNW"):
        tweet_query = get_keywords_sota(df_query)
    if(u_keyword_extractor=="YAKE"):
        tweet_query = get_keywords_yake(tweet_query)
    if(u_keyword_extractor=="RAKE"):
        tweet_query = get_keywords_rake(tweet_query)
    if(u_keyword_extractor=="TextRank"):
        tweet_query = get_keywords_text_rank(tweet_query)
    if(u_keyword_extractor=="KeyBert"):
        tweet_query = get_keywords_keybert(tweet_query)


    docs_preprocessed, total_documents = get_docs_preprocessed(u_present_date, u_prev_date, u_next_date)
    if(u_model=='jaccard'):
        return jaccard_results(docs_preprocessed, tweet_query, u_base_hashtag, u_time, u_location)
    if(u_model=='cosine_count'):
        return cosine_count_results(docs_preprocessed, tweet_query, u_base_hashtag, u_time, u_location)
    if(u_model == 'cosine_tf'):
        return cosine_tf_results(docs_preprocessed, tweet_query, u_base_hashtag, u_time, u_location)
    if(u_model == 'bip'):
        return bip_results(tweet_query, u_base_hashtag, u_time, u_location)
    if(u_model == 'bm25'):
        return bm25_results(docs_preprocessed, tweet_query, u_base_hashtag, u_time, u_location)
    if(u_model == 'soft_cosine'):
        return cosine_tf_results(docs_preprocessed, tweet_query, u_base_hashtag, u_time, u_location)
    if(u_model == 'best_metric'):
        return bm25_results(docs_preprocessed, tweet_query, u_base_hashtag, u_time, u_location)
    if(u_model == 'tf_idf1'):
        if(u_keyword_extractor=='none'):
            return tf_idf_1(docs_preprocessed, tweet_query, u_base_hashtag, u_time, u_location, True, total_documents)
        else:
            return tf_idf_1(docs_preprocessed, tweet_query, u_base_hashtag, u_time, u_location, False, total_documents)
    if(u_model == 'tf_idf2'):
        if(u_keyword_extractor=='none'):
            return tf_idf_2(docs_preprocessed, tweet_query, u_base_hashtag, u_time, u_location, True, total_documents)
        else:
            return tf_idf_2(docs_preprocessed, tweet_query, u_base_hashtag, u_time, u_location, False, total_documents)

    if(u_model == 'tf_idf3'):
        if(u_keyword_extractor=='none'):
            return tf_idf_3(docs_preprocessed, tweet_query, u_base_hashtag, u_time, u_location, True, total_documents)
        else:
            return tf_idf_3(docs_preprocessed, tweet_query, u_base_hashtag, u_time, u_location, False, total_documents)

    if(u_model == 'tf_idf4'):
        if(u_keyword_extractor=='none'):
            return tf_idf_4(docs_preprocessed, tweet_query, u_base_hashtag, u_time, u_location, True, total_documents)
        else:
            return tf_idf_4(docs_preprocessed, tweet_query, u_base_hashtag, u_time, u_location, False, total_documents)

    if(u_model == 'tf_idf5'):
        if(u_keyword_extractor=='none'):
            return tf_idf_5(docs_preprocessed, tweet_query, u_base_hashtag, u_time, u_location, True, total_documents)
        else:
            return tf_idf_5(docs_preprocessed, tweet_query, u_base_hashtag, u_time, u_location, False, total_documents)
