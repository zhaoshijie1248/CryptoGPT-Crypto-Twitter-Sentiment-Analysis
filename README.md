
# CryptoGPT: Crypto Twitter Sentiment Analysis

Welcome to CryptoGPT! This project combines Streamlit, ChatGPT, and LangChain to analyze the sentiment of tweets related to cryptocurrencies. By utilizing Streamlit, we'll create a user-friendly interface that allows us to interact with our sentiment analysis application effortlessly.

## Project Setup

We'll use Python 3.11.3 for this project, and the directory structure will be as follows:

### Libraries

Install the required libraries:

```bash
pip install streamlit langchain tweety plotly
```

### Config

We'll use `black` and `isort` for formatting and import sorting. Additionally, we'll configure VSCode for the project.

### Streamlit

Streamlit is an open-source Python library designed for building custom web applications with ease. It allows us to create interactive and visually appealing data-driven applications using Python. With Streamlit, we can quickly transform our data analysis code into shareable web applications, making it ideal for our sentiment analysis project. Let's leverage the power of Streamlit to create a seamless and user-friendly interface for analyzing the sentiment of cryptocurrency tweets.

## Get Tweets

To fetch tweets for our analysis, we'll make use of the `tweety` library. This library interacts with Twitter's frontend API to retrieve the desired tweets.

### Clean Tweets

We can remove unnecessary elements like URLs, new lines, and multiple spaces from the tweets, as they are not relevant for our sentiment analysis and will save tokens for ChatGPT.

```python
import re

def clean_tweet(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'
', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
```

### Create DataFrame

We'll use a dataframe to organize and easily visualize the tweets.

```python
import pandas as pd

def create_dataframe_from_tweets(tweets):
    df = pd.DataFrame([{'date': tweet.date, 'author': tweet.author, 'text': clean_tweet(tweet.text), 'views': tweet.views} for tweet in tweets])
    return df
```

## Sentiment Analysis

We'll use ChatGPT and LangChain to analyze the sentiment of the tweets.

```python
from langchain.llms import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def analyze_sentiment(tweets, twitter_handle):
    llm = ChatOpenAI(model="text-davinci-002")
    prompt = PromptTemplate("Analyze the sentiment of the following tweets from {twitter_handle}: {tweets}")
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run({"twitter_handle": twitter_handle, "tweets": tweets})
    sentiment = json.loads(response)
    return sentiment
```

## Visualize Sentiment

We'll utilize Plotly to visualize the sentiment. We can generate a line chart to visualize the sentiment trends. Additionally, we'll display a dataframe that contains the sentiment data.

```python
import plotly.express as px

def plot_sentiment(df):
    fig = px.line(df, x='date', y='sentiment', title='Sentiment Over Time')
    fig.show()
```

## Streamlit App

Finally, we'll create the Streamlit app to interact with our sentiment analysis pipeline.

```python
import streamlit as st

st.title("Crypto Twitter Sentiment Analysis")
twitter_handle = st.text_input("Enter Twitter Handle", "")
if twitter_handle:
    tweets = get_tweets(twitter_handle)
    df = create_dataframe_from_tweets(tweets)
    sentiment = analyze_sentiment(tweets, twitter_handle)
    df['sentiment'] = df['date'].map(sentiment)
    plot_sentiment(df)
    st.write(df)
```

## Conclusion

In this tutorial, we covered the process of sentiment analysis on cryptocurrency tweets using LangChain and ChatGPT. We learned how to download and preprocess tweets, visualize sentiment data using Plotly, and create a Streamlit application to interact with the sentiment analysis pipeline.

The integration of Streamlit allows us to create an interactive and intuitive interface for users to input Twitter handles, view sentiment analysis results, and visualize the sentiment trends over time.
