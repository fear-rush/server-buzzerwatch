import re
import random

import numpy as np
import pandas as pd
import tweepy as tw
import os 

import torch
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm
from dotenv import load_dotenv


from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from nltk.tokenize import TweetTokenizer

from function import twitter_access, twitter_api, tweets_fetching, remove_emojis, text_cleaning, set_seed, count_param, get_lr, metrics_to_string
from data_utils import DocumentSentimentDataLoader, DocumentSentimentDataset

def sentiment (tweet):
    #1. Initialize model
    TOKENIZER_PATH = './models/tokenizer'
    CONFIG_PATH = './models/config'
    MODEL_PATH = './models/model'

    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
    config = BertConfig.from_pretrained(CONFIG_PATH)
    config.num_labels = DocumentSentimentDataset.NUM_LABELS

    model = BertForSequenceClassification.from_pretrained(MODEL_PATH, config=config)
    w2i, i2w = DocumentSentimentDataset.LABEL2INDEX, DocumentSentimentDataset.INDEX2LABEL

    #2. Twitter access
    # twitter_account = pd.read_csv('Twitter Developer Account.csv')
    # consumer_api_key, consumer_api_key_secret, access_token, access_token_secret = twitter_access(twitter_account)
    # api = twitter_api(consumer_api_key, consumer_api_key_secret, access_token, access_token_secret)
    consumer_api_key = os.getenv("CONSUMER_API_KEY")
    consumer_api_key_secret = os.getenv("CONSUMER_API_KEY_SECRET")
    access_token = os.getenv("ACCESS_TOKEN")
    access_token_secret = os.getenv("ACCESS_TOKEN_SECRET")
    api = twitter_api(consumer_api_key, consumer_api_key_secret, access_token, access_token_secret)

    #3. Take tweets
    raw_tweets_list = tweets_fetching(api, tweet, 100) # kata kuncinya bisa diubah, jumlah tweet yang mau diambil juga bisa diubah

    #4. Save tweets
    tweets_df = pd.DataFrame({
        'raw' : raw_tweets_list
    })

    #5. Clean tweets
    tweets_df['clean'] = tweets_df['raw'].apply(lambda x: text_cleaning(x))

    #6. Take sentiment
    for ind in tweets_df.index:
        text = tweets_df['clean'][ind]
        subwords = tokenizer.encode(text)
        subwords = torch.LongTensor(subwords).view(1, -1).to(model.device)

        logits = model(subwords)[0]
        label = torch.topk(logits, k=1, dim=-1)[1].squeeze().item()

        tweets_df.loc[ind,'sentiment'] = i2w[label]
        tweets_df.loc[ind,'percentage'] = float("{:.2f}".format(F.softmax(logits, dim=-1).squeeze()[label] * 100))

    #7. Display tweets
    print(tweets_df)
    return tweets_df