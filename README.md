# Financial News Sentiment Analysis

## Description

The Financial News Sentiment Analysis project aims to investigate the predictive power of sentiment analysis on financial news articles in forecasting stock price movements.
By analyzing sentiment scores derived from financial news, this project seeks to provide insights into the correlation between market sentiment and stock price fluctuations.

## Table of Contents

1. [Installation]
2. [Usage]
3. [Contributing]
4. [License]
5. [Contact Information)

## Installation
1. Install the required libraries

## Usage

Follow these steps to use the project:

1. **Data Collection**

3. **Data Preprocessing**

4. **Sentiment Analysis**

import pandas as pd
import requests
import torch
import time
from bs4 import BeautifulSoup
from datetime import datetime
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from finvizfinance.quote import finvizfinance
from finvader import finvader
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import BertTokenizer, BertForSequenceClassification

# Function to preprocess text content
def preprocess_text(content):
    # Convert to lowercase
    content = content.lower()
    # Remove punctuation and special characters
    content = re.sub(r'[^a-zA-Z\s]', '', content)
    # Tokenize
    tokens = word_tokenize(content)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join tokens back into string
    processed_content = ' '.join(tokens)
    return processed_content

# Function to fetch article content from URL
def fetch_article_content(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        # Check if content type is HTML
        if 'text/html' in response.headers['Content-Type']:
            soup = BeautifulSoup(response.content, 'html.parser')
            # Attempt to find main content using different strategies based on the structure of the page
            main_content = None
            # Look for paragraphs
            paragraphs = soup.find_all('p')
            if paragraphs:
                main_content = ' '.join([para.get_text() for para in paragraphs])
            # Look for specific divs or sections containing content
            if not main_content:
                main_content_div = soup.find('div', {'class': 'bw-release-story'})
                if main_content_div:
                    main_content = main_content_div.get_text(separator=' ')
            # Fallback to getting all text in the page
            if not main_content:
                main_content = soup.get_text(separator=' ')
            # Preprocess content
            processed_content = preprocess_text(main_content)

            return processed_content
        else:
            print(f"Unsupported content type for URL: {url}")
            return None

    except requests.exceptions.HTTPError as http_err:
        return None
    except requests.exceptions.RequestException as req_err:
        return None
    except Exception as e:
        return None

def fetch_news_content(tickers, start_date, end_date):
    all_news = []  # List to store news articles
    for ticker in tickers:
        try:
            # Initialize finvizfinance with ticker
            stock = finvizfinance(ticker)
            # Fetch ticker news metadata
            news_df = stock.ticker_news()
            # Current date for handling 'Today'
            today_date = datetime.now()
            # Convert 'Date' column to datetime objects, handle 'Today' label
            def parse_date(date_str):
                if date_str == 'Today':
                    return today_date
                try:
                    return pd.to_datetime(date_str, format='%b-%d-%y %I:%M%p', errors='coerce')
                except ValueError:
                    return None
            news_df['Date'] = news_df['Date'].apply(parse_date)
            # Drop rows where 'Date' could not be parsed
            news_df = news_df.dropna(subset=['Date'])
            # Filter news by date range
            news_df = news_df[(news_df['Date'] >= start_date) & (news_df['Date'] <= end_date)].copy()
            # Fetch content for each news article
            news_df.loc[:, 'Processed_Content'] = news_df['Link'].apply(fetch_article_content)
            # Filter out rows where content could not be fetched
            news_df = news_df[news_df['Processed_Content'].notnull()].copy()
            # Add ticker column
            news_df['Ticker'] = ticker
            # Append to all_news list
            all_news.append(news_df)
        except Exception as e:
            print(f"Error fetching news for {ticker}: {e}")
    # Concatenate all news articles into a single DataFrame
    if all_news:
        combined_news_df = pd.concat(all_news, ignore_index=True)
    else:
        combined_news_df = pd.DataFrame()
    return combined_news_df

# Function to compute sentiment score using finvader
def compute_sentiment_score_finvader(text):
    sentiment_score = finvader(text, use_sentibignomics=True, use_henry=True, indicator='compound')
    return sentiment_score
# Function to compute sentiment score using NLTK's Sentiment Intensity Analyzer
def compute_sentiment_score_nltk(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)['compound']
    return sentiment_score
#Function to compute sentiment score using FinBERT
def compute_sentiment_score_finroberta(text, tokenizer, model):
    tokens = tokenizer.encode(text, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
    with torch.no_grad():
        output = model(tokens)
    logits = output.logits
    probabilities = torch.softmax(logits, dim=1).numpy()[0]
    # Get sentiment score
    negative_score = probabilities[0]
    neutral_score = probabilities[1]
    positive_score = probabilities[2]
    # The sentiment score can be a weighted sum
    sentiment_score = positive_score - negative_score  # value between -1 & 1
    return sentiment_score

# Read tickers and percentage changes from CSV file
def get_all_tickers_and_changes(csv_filename):
    df = pd.read_csv(csv_filename)

    # Select all tickers and their percentage changes
    tickers_and_changes = df[['Ticker', 'Change']].copy()

    return tickers_and_changes

# Function to fetch stock data from the given API URL with retry logic
def fetch_stock_data(api_url, max_retries=5, backoff_factor=1):
    attempt = 0
    while attempt < max_retries:
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            return response.content
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:  # Too Many Requests
                attempt += 1
                wait_time = backoff_factor * (2 ** (attempt - 1))  # Exponential backoff
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise e
    raise Exception("Max retries exceeded")

# Main function
def main():
    # API URL with specific filters and API token
    api_url = "https://elite.finviz.com/export.ashx?v=151&p=i1&f=cap_0.01to,geo_usa|china|france|europe|australia|belgium|canada|chinahongkong|germany|hongkong|iceland|japan|newzealand|ireland|netherlands|norway|singapore|southkorea|sweden|taiwan|unitedarabemirates|unitedkingdom|switzerland|spain,sh_curvol_o100,sh_relvol_o2&ft=4&o=-change&ar=10&auth=28943bdd-1830-4a3d-a176-1293e3bc2f4e"

    try:
        # Fetch stock data with retry logic
        data = fetch_stock_data(api_url)
        if data:
            with open("finviz.csv", "wb") as f:
                f.write(data)
            print("CSV file created successfully: finviz.csv")
        else:
            print("No data fetched.")
        # Use the newly created CSV file to get all tickers and their percentage changes
        csv_filename = 'finviz.csv'
        tickers_and_changes = get_all_tickers_and_changes(csv_filename)
        # Extract tickers and changes
        tickers = tickers_and_changes['Ticker'].tolist()
        changes = tickers_and_changes['Change'].tolist()
        # Print tickers that will be used for output file
        print(f"Tickers being used: {tickers}")
        #Set desired start date
        start_date = datetime(2024, 7, 27)
        end_date = datetime.now()
        news_df = fetch_news_content(tickers, start_date, end_date)
        if not news_df.empty:
            # Initialize FinBert model and tokenizer
            model_name = 'yiyanghkust/finbert-tone'
            tokenizer = BertTokenizer.from_pretrained(model_name)
            model = BertForSequenceClassification.from_pretrained(model_name)
            news_df['SS_Finvader'] = news_df['Processed_Content'].apply(compute_sentiment_score_finvader)
            news_df['SS_NLTK'] = news_df['Processed_Content'].apply(compute_sentiment_score_nltk)
            news_df['SS_FinRoberta'] = news_df['Processed_Content'].apply(lambda x: compute_sentiment_score_finroberta(x, tokenizer, model))

            # Map percentage changes to news_df
            news_df = news_df.merge(tickers_and_changes, left_on='Ticker', right_on='Ticker', how='left')

            # Output CSV file
            csv_filename = f'news_{start_date.strftime("%Y-%m-%d")}_to_{end_date.strftime("%Y-%m-%d")}.csv'
            news_df.to_csv(csv_filename, index=False, columns=['Date', 'Ticker', 'Title', 'Link', 'Processed_Content', 'SS_Finvader', 'SS_NLTK', 'SS_FinRoberta', 'Change'])
            print(f"CSV file '{csv_filename}' has been created.")
        else:
            print("No news data fetched.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == '__main__':
    main()

5. **Box Plot**

6. **Evaluation Metrics**


## Contributing

We welcome contributions to enhance this project! Here’s how you can contribute:

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Commit your changes
5. Push to the branch 
6. Open a pull request

Please ensure your code adheres to our coding standards and include tests where applicable.

## Contact Information

For any questions or suggestions, please contact the project maintainers:

- [Dongyeon Kang](mailto:danny379k@gmail.com)

You can also create an issue on the [IST495](https://github.com/eastkite00/IST495/issues).
