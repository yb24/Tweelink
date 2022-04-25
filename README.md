# Tweelink

By **Amisha Aggarwal**, **Aryan Behal**, **Harman Singh**, **Mayank Gupta**, **Yash Bhargava** and **Yash Tanwar** Information Retrieval (CSE508) under the guidance of **Dr. Rajiv Ratn Shah** from **Indraprastha Institute of Information Technology, Delhi**.

## Introduction
- ### Motivation and Problem Statement
<p align="justify">Twitter is an extremely popular microblogging and social networking site launched in 2006. Today it boasts of hundreds of millions of monthly active users. Users can post, like, tweet and retweet 280 characters at a time. No other social platform comes close to capturing the pulse of the world in real time. Twitter users comprise people from all walks of life. Equally rich is the diversity of topics tweeted by users every second of the day. While a tweet is representative of a person’s thought process, a hashtag is the culmination of collective thought of the populace. While the tweets shape the trend of a hashtag, the hashtag shapes the conversation as well, making a self feeding loop.</p>  
<p align="justify">For better or for worse, Twitter plays a key role in shaping the conversation on topics that affect everyone’s lives. From coronavirus to anti-vaccination protests to the Russia-Ukraine crisis, Twitter makes its presence felt everywhere. Our project aims to capture the sentiment and context of a hashtag and link it to relevant news articles. The context behind a hashtag is clouded by its informal structure. A hashtag is generally an amalgamation or portmanteau of several words making it difficult to decipher. This will help the uninformed wade through countless tweets to understand the context of the trend while combating misinformation at the same time.</p>

- ### Novelty
<p align="justify">The novelty of our project lies in its uniqueness and ability to retrieve information out of a hashtag.To the best of our knowledge, there isn’t any existing research or technology attempting to link hashtags to news articles. Some existing papers have tried to use the tweets containing URLs, but we are not dependent only on URLs within the tweet. Our novelty lies in the fact that we consider external news articles explaining the trending news/controversy. We face the challenge of working with a corpus of extremely short texts on which regular NLP techniques may yield poor results. The text itself may be informal, multi-lingual and full of grammatical errors.</p>

## Dataset and Models
Link to dataset, pickled models : [click here](https://drive.google.com/drive/folders/1mPMliffHTBokYxBhK-4xPJ5c29NxIJAT?usp=sharing)

## Models
- Jaccard Similarity Ranked Retrieval
- TF-IDF (Weighing Scheme: Binary, Raw Count, Term Frequency, Log Normalization, Double Normalization) Ranked Retrieval
- Cosine Similarity (Count Vectorization & TF-IDF Vectorization) Ranked Retrieval
- Binary Independence Model

## Repository Description
- ### Baseline Results
  Code files for baseline models
- ### EDA
  Code for Exploratory Data Analysis
- ### Miscellaneous
  Heplful Links
- ### Preprocessing
  Code for preprocessing Tweets and Articles Dataset
- ### Reports
  Proposal and Mid-Report
- ### Twitter-Data-Collection
  - #### Operator Precedence
    Operator Precedence of Twitter API v2
  - #### Test Code
    Sample codes for Twitter API v2
- ### Web
  Code for frontend, backend, db

## Contact
For further queries feel free to reach out to following contributors.  
Amisha Aggarwal (amisha19016@iiitd.ac.in)  
Aryan Behal (aryan19026@iiitd.ac.in)  
Harman Singh (harman19042@iiitd.ac.in)  
Mayank Gupta (mayank19059@iiitd.ac.in)  
Yash Bhargava (yash19289@iiitd.ac.in)  
Yash Tanwar (yash19130@iiitd.ac.in)  
