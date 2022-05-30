import tweet_config
import tweepy
import pandas as pd
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# API keys you'll receive when you register for a Twitter developer account and create your first app
api_key = tweet_config.api_key
api_secret = tweet_config.api_key_secret

access_token = tweet_config.access_token
access_token_secret = tweet_config.access_token_secret

auth = tweepy.OAuthHandler(api_key, api_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

# Creates a Listener class to handle streaming tweets and
# limits the number of tweets returned to 100


class Listener(tweepy.Stream):
    tweets = []
    limit = 100

    def on_status(self, status):
        self.tweets.append(status)

        if len(self.tweets) == self.limit:
            self.disconnect()


stream_tweet = Listener(api_key, api_secret, access_token, access_token_secret)

# Keywords you want to return

keywords = ['#MemorialDay']

stream_tweet.filter(track=keywords)

columns = ['User', 'Tweet']
data = []

for tweet in stream_tweet.tweets:
    if not tweet.truncated:
        data.append([tweet.user.screen_name, tweet.text])
    else:
        data.append([tweet.user.screen_name, tweet.extended_tweet['full_text']])


df = pd.DataFrame(data=data, columns=columns)

sia = SentimentIntensityAnalyzer()


def sentiment_score(vector):
    return sia.polarity_scores(vector)


df['scores'] = df['Tweet'].apply(sentiment_score)

df_scores = pd.json_normalize(df['scores'])

final_df = df.merge(df_scores, right_index=True, left_index=True)
final_df = final_df[['User', 'Tweet', 'neg', 'neu', 'pos', 'compound']]

# Exports final dataframe to a local CSV file
final_df.to_csv('final_tweets.csv', index=False)
