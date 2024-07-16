
import os
from datetime import datetime
from typing import Dict

import pandas as pd
import plotly.express as px
import streamlit as st
from tweety import Twitter

from sentiment_analyzer import analyze_sentiment, create_dataframe_from_tweets

twitter_client = Twitter("session")

def on_add_author():
    twitter_handle = st.session_state.twitter_handle
    if twitter_handle.startswith("@"):
        twitter_handle = twitter_handle[1:]
    if twitter_handle in st.session_state.twitter_handles:
        return
    all_tweets = twitter_client.get_tweets(twitter_handle)
    if len(all_tweets) == 0:
        return
    latest_30_tweets = all_tweets[:30]
    st.session_state.twitter_handles[twitter_handle] = all_tweets[0].author.name
    st.session_state.tweets.extend(latest_30_tweets)
    st.session_state.author_sentiment[twitter_handle] = analyze_sentiment(
        twitter_handle, latest_30_tweets
    )
    # st.write(f"Author Sentiment for {twitter_handle}: {st.session_state.author_sentiment[twitter_handle]}")

def create_sentiment_dataframe(tweets: list) -> pd.DataFrame:
    # Collect dates from the latest 30 tweets
    dates = sorted({tweet.date for tweet in tweets}, reverse=True)[:30]
    dates = sorted(dates)  # Sort again to ensure proper order
    index_range = list(range(1, len(dates) + 1))
    chart_data = {"index": index_range}

    # Collect sentiment data
    sentiment_data = st.session_state.author_sentiment
    for author, sentiments in sentiment_data.items():
        author_sentiment = []
        for date in dates:
            str_date = date.strftime("%Y-%m-%d")
            if str_date in sentiments:
                author_sentiment.append(sentiments[str_date])
           
        chart_data[author] = author_sentiment

    sentiment_df = pd.DataFrame(chart_data)
    sentiment_df.set_index("index", inplace=True)

    if not sentiment_df.empty:
        sentiment_df["Overall"] = sentiment_df.mean(skipna=True, axis=1)
        
    # Ensure all columns are numeric
    sentiment_df = sentiment_df.apply(pd.to_numeric, errors='coerce')
    # st.write("Sentiment DataFrame:", sentiment_df)

    return sentiment_df

st.set_page_config(
    layout="wide",
    page_title="CryptoGPT: Crypto Twitter Sentiment Analysis",
    page_icon="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/1f4c8.png",
)

st.markdown(
    "<h1 style='text-align: center'>CryptoGPT: Crypto Twitter Sentiment Analysis</h1>",
    unsafe_allow_html=True,
)

if not "tweets" in st.session_state:
    st.session_state.tweets = []
    st.session_state.twitter_handles = {}
    st.session_state.api_key = ""
    st.session_state.author_sentiment = {}

os.environ["OPENAI_API_KEY"] = st.session_state.api_key

col1, col2 = st.columns(2)

with col1:
    st.text_input(
        "OpenAI API Key",
        type="password",
        key="api_key",
        placeholder="sk-...4242",
        help="Get your API key: https://platform.openai.com/account/api-keys",
    )

    with st.form(key="twitter_handle_form", clear_on_submit=True):
        st.subheader("Add Twitter Accounts", anchor=False)
        st.text_input(
            "Twitter Handle", value="", key="twitter_handle", placeholder="@saylor"
        )
        submit = st.form_submit_button(label="Add Tweets", on_click=on_add_author)

    if st.session_state.twitter_handles:
        st.subheader("Twitter Handles", anchor=False)
        for handle, name in st.session_state.twitter_handles.items():
            handle = "@" + handle
            st.markdown(f"{name} ([{handle}](https://twitter.com/{handle}))")

    st.subheader("Tweets", anchor=False)

    st.dataframe(
        create_dataframe_from_tweets(st.session_state.tweets), use_container_width=True
    )

with col2:
    sentiment_df = create_sentiment_dataframe(st.session_state.tweets)
    if not sentiment_df.empty:
        fig = px.line(
            sentiment_df,
            x=sentiment_df.index,
            y=sentiment_df.columns,
            labels={"index": "Date", "value": "Sentiment"},
        )
        fig.update_layout(yaxis_range=[0, 100])
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

        st.dataframe(sentiment_df, use_container_width=True)
