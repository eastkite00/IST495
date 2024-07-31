#Introduction

**Importance of Sentiment Analysis in the Financial Market** <br>
The financial market is highly sensitive to news and public sentiment. Market participants rely on news to make informed decisions about buying or selling stocks. Positive news about a company, such as strong earnings reports or new product launches, can drive stock prices up, while negative news, like legal troubles or poor financial performance, can push stock prices down. Thus, accurately gauging the sentiment of financial news can provide valuable insights and enhance trading strategies. In addition, analyzing sentiment helps in understanding these behavioral aspects, providing a more comprehensive view of market dynamics.

**Purpose** <br>
The primary goal of the research is to investigate the predictive power of sentiment analysis on financial news articles in forecasting stock price movements. By comparing different sentiment analysis models, I aim to identify each model's features in capturing the nuances of market sentiment and their impact on stock prices.

#Project Overview

In this project, I aim to explore the correlation between sentiment scores derived from financial news articles and the subsequent movement in stock prices. The project involves several key steps:


1.   **Literature Review/Case Study:**
existing research and case studies will be examined to identify the theoretical foundations, methodologies, and findings that have shaped the current understanding of how sentiment analysis can impact the stock price movements
2.   **Data Collection:**  
financial news articles and relevant stock price data will be sourced from finviz financial news websites and databases using API
3.   **Data Preprocessing:**
cleaning and preparing the collected data for analysis is involved in this step and also creating a structured dataset where each news article is linked to the corresponding stock price movement
4.   **Sentiment Analysis Techniques Implementation:**
applying various sentiment analysis models to the preprocessed news articles to compute sentiment scores. FinVader, NLTK's Sentiment Intensity Analyzer, and FinBERT will be used to generate sentiment scores for each article.
5.   **Evaluation:**
robustness of each sentiment analysis model in stock market prediction by comparing the sentiment scores with actual stock price changes using proper metrics. Any visualization tools will be utilized to illustrate the finding from generated csv file.

Resources & Tools

*   YouTube Videos to get reference on methods of using necessary libraries and implementing different sentiment analysis models.
*   Effectively utilized LLM such as ChatGPT to figure out the errors found in coding process







#Literature Review

https://doaj.org/article/b289000ac68844f2948c700f5715dadc

**Objective** <br>
The paper aims to enhance stock price prediction by incorporating sentiment analysis into regression models. By leveraging financial news headlines, the study seeks to improve the accuracy of stock price forecasts through advanced sentiment analysis techniques.

**Methodology** <br>
Utilized the VADER model to generate daily sentiment scores from financial news headlines provided by FinViz. Three types of regression models were employed: linear, quadratic, and cubic autoregressions.
The sentiment score was integrated into these models to assess its impact on the goodness of fit. Ordinary least squares regression was performed using SPSS, with equal weight aggregation for all data points. Also, the study focused on companies from the S&P 500 index and sentiment data was collected from August to September 2022, covering 37 days.

**Key Findings** <br>
The average sentiment score in the news was found to be 0.06 with a volatility of 0.18. Polynomial autoregressions demonstrated a higher R-squared value compared to linear autoregressions, indicating a better fit. The study concluded that incorporating sentiment analysis as an outside factor in regression models can significantly enhance the prediction accuracy of stock prices.

#Installation

Installed necessary libraries for execution of codes. Impored the nltk library and download specific datasets required for various NLP tasks. The libraries and datasets are used for web scraping, data parsing, and sentiment analysis in financial contexts. The nltk downloads ensure that the required resources are available for text preprocessing and sentiment analysis tasks.


```python
!pip install finvizfinance
!pip install requests
!pip install beautifulsoup4
!pip install nltk
!pip install finvader
!pip install transformers
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('vader_lexicon')
```

    Collecting finvizfinance
      Downloading finvizfinance-1.0.1-py3-none-any.whl.metadata (5.0 kB)
    Requirement already satisfied: pandas in /usr/local/lib/python3.10/dist-packages (from finvizfinance) (2.0.3)
    Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from finvizfinance) (2.31.0)
    Requirement already satisfied: beautifulsoup4 in /usr/local/lib/python3.10/dist-packages (from finvizfinance) (4.12.3)
    Requirement already satisfied: lxml in /usr/local/lib/python3.10/dist-packages (from finvizfinance) (4.9.4)
    Requirement already satisfied: soupsieve>1.2 in /usr/local/lib/python3.10/dist-packages (from beautifulsoup4->finvizfinance) (2.5)
    Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.10/dist-packages (from pandas->finvizfinance) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas->finvizfinance) (2024.1)
    Requirement already satisfied: tzdata>=2022.1 in /usr/local/lib/python3.10/dist-packages (from pandas->finvizfinance) (2024.1)
    Requirement already satisfied: numpy>=1.21.0 in /usr/local/lib/python3.10/dist-packages (from pandas->finvizfinance) (1.25.2)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->finvizfinance) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->finvizfinance) (3.7)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->finvizfinance) (2.0.7)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->finvizfinance) (2024.7.4)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.2->pandas->finvizfinance) (1.16.0)
    Downloading finvizfinance-1.0.1-py3-none-any.whl (44 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m44.1/44.1 kB[0m [31m2.9 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: finvizfinance
    Successfully installed finvizfinance-1.0.1
    Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (2.31.0)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests) (3.7)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests) (2.0.7)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests) (2024.7.4)
    Requirement already satisfied: beautifulsoup4 in /usr/local/lib/python3.10/dist-packages (4.12.3)
    Requirement already satisfied: soupsieve>1.2 in /usr/local/lib/python3.10/dist-packages (from beautifulsoup4) (2.5)
    Requirement already satisfied: nltk in /usr/local/lib/python3.10/dist-packages (3.8.1)
    Requirement already satisfied: click in /usr/local/lib/python3.10/dist-packages (from nltk) (8.1.7)
    Requirement already satisfied: joblib in /usr/local/lib/python3.10/dist-packages (from nltk) (1.4.2)
    Requirement already satisfied: regex>=2021.8.3 in /usr/local/lib/python3.10/dist-packages (from nltk) (2024.5.15)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (from nltk) (4.66.4)
    Collecting finvader
      Downloading finvader-1.0.4-py3-none-any.whl.metadata (4.5 kB)
    Collecting nltk==3.6.2 (from finvader)
      Downloading nltk-3.6.2-py3-none-any.whl.metadata (2.9 kB)
    [33mWARNING: Package 'nltk' has an invalid Requires-Python: Invalid specifier: '>=3.5.*'[0m[33m
    [0mRequirement already satisfied: click in /usr/local/lib/python3.10/dist-packages (from nltk==3.6.2->finvader) (8.1.7)
    Requirement already satisfied: joblib in /usr/local/lib/python3.10/dist-packages (from nltk==3.6.2->finvader) (1.4.2)
    Requirement already satisfied: regex in /usr/local/lib/python3.10/dist-packages (from nltk==3.6.2->finvader) (2024.5.15)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (from nltk==3.6.2->finvader) (4.66.4)
    Downloading finvader-1.0.4-py3-none-any.whl (45 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m45.2/45.2 kB[0m [31m2.7 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nltk-3.6.2-py3-none-any.whl (1.5 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.5/1.5 MB[0m [31m22.2 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: nltk, finvader
      Attempting uninstall: nltk
        Found existing installation: nltk 3.8.1
        Uninstalling nltk-3.8.1:
          Successfully uninstalled nltk-3.8.1
    Successfully installed finvader-1.0.4 nltk-3.6.2
    Requirement already satisfied: transformers in /usr/local/lib/python3.10/dist-packages (4.42.4)
    Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from transformers) (3.15.4)
    Requirement already satisfied: huggingface-hub<1.0,>=0.23.2 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.23.5)
    Requirement already satisfied: numpy<2.0,>=1.17 in /usr/local/lib/python3.10/dist-packages (from transformers) (1.25.2)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from transformers) (24.1)
    Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.10/dist-packages (from transformers) (6.0.1)
    Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.10/dist-packages (from transformers) (2024.5.15)
    Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from transformers) (2.31.0)
    Requirement already satisfied: safetensors>=0.4.1 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.4.3)
    Requirement already satisfied: tokenizers<0.20,>=0.19 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.19.1)
    Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.10/dist-packages (from transformers) (4.66.4)
    Requirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub<1.0,>=0.23.2->transformers) (2024.6.1)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub<1.0,>=0.23.2->transformers) (4.12.2)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (3.7)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (2.0.7)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (2024.7.4)


    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Unzipping corpora/stopwords.zip.
    [nltk_data] Downloading package punkt to /root/nltk_data...
    [nltk_data]   Unzipping tokenizers/punkt.zip.
    [nltk_data] Downloading package wordnet to /root/nltk_data...
    [nltk_data] Downloading package vader_lexicon to /root/nltk_data...





    True



#Functions

**preprocess_text** <br>

*   Converts the text to lowercase to ensure uniformity.
*   Removes punctuation and special characters, retaining only alphabetic characters and whitespace.
*   Tokenizes the text into individual words.
*   Removes common English stopwords to reduce noise.
*   Lemmatizes each token to reduce words to their base or root form.
*   Joins the tokens back into a single string and returns the processed content.

**fetch_article_content** <br>

*   Sends an HTTP GET request to the provided URL using a user-agent header to mimic a browser.
*   Parses the HTML content using BeautifulSoup.
*   Attempts to extract the main content by looking for paragraph tags or specific divs.
*   If specific tags are not found, extracts all text from the page.
*   Preprocesses the extracted text using 'preprocess_text' function and returns it.

**fetch_news_content** <br>

*   Iterates over the list of tickers.
*   Converts the 'Date' column in the news DataFrame to datetime objects, handling 'Today' labels appropriately.
*   Filters the news articles to only include those within the specified date range.
*   Fetches and preprocesses the content for each news article using function 'fetch_article_content'.
*   Adds the processed content and ticker information to the DataFrame.
*    Combinesall individual news DataFrames into a single DataFrame and returns it.

**compute_sentiment_score_finvader** <br>

*   Uses FinVADER with specific settings to compute the compound sentiment score.


**compute_sentiment_score_nltk** <br>
*   Initializes the Sentiment Intensity Analyzer from NLTK.
*   Computes the polarity scores for the text.
*   Extracts and returns the compound score

**compute_sentiment_score_finroberta** <br>
*   Encodes the text into tokens using the BERT tokenizer, ensuring truncation and padding to the maximum length of 512 tokens.
*   Passes the tokens through the BERT model to get the output logits.
*   Applies the softmax function to the logits to get probabilities for each sentiment class
*   Computes the sentiment score as the difference between positive and negative probabilities.
*  Returns the sentiment score, which ranges from -1 (strongly negative) to 1 (strongly positive).


**get_all_tickers_and_changes** <br>
*   Selects the 'Ticker' and 'Change' columns from finviz.csv file


**fetch_stock_data** <br>
*   Attempts to fetch the data from the API URL.
*   If a 429 (Too Many Requests) error occurs, retries the request with exponential backoff.
*   Raises an exception if the maximum number of retries is exceeded.


```python
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

```

    CSV file created successfully: finviz.csv
    Tickers being used: ['BFI', 'YIBO', 'CREV', 'COUR', 'NWL', 'EVRI', 'AEYE', 'RGC', 'HUDI', 'LOOP', 'ESLA', 'DOUG', 'SGMO', 'LUXH', 'MMM', 'PSNL', 'KARO', 'MHK', 'IGT', 'NXGL', 'KNSL', 'LUMO', 'CHTR', 'WBUY', 'CABO', 'LFVN', 'TCS', 'OCEA', 'CMAX', 'ASTS', 'CGTX', 'CLLS', 'AVTR', 'COLB', 'BMY', 'NSC', 'RR', 'ABTS', 'WF', 'ZPTA', 'MEDS', 'SSTI', 'CBAN', 'FBIN', 'SERV', 'ERIE', 'GTX', 'CNC', 'HBI', 'CVM', 'CUZ', 'APLM', 'ORN', 'SWIM', 'TZOO', 'CLMT', 'MIRM', 'SATL', 'LW', 'CYH', 'FTAI', 'SWK', 'WT', 'DECK', 'ALEX', 'NOV', 'SSNC', 'KB', 'BIVI', 'DDC', 'HHH', 'DOC', 'VLTO', 'EME', 'SRCE', 'HYZN', 'RYAAY', 'CNTA', 'PFBC', 'TYL', 'HNRA', 'EW', 'UPC', 'AOMR', 'DK', 'UHS', 'KROS', 'SLRN', 'ALSN', 'BFST', 'DOYU', 'TECK', 'PRG', 'NCNO', 'OPBK', 'ATR', 'BKE', 'IMAX', 'NVRI', 'MOH', 'EBC', 'LQDT', 'KITT', 'SWN', 'CBRE', 'UL', 'PFC', 'WRN', 'INVH', 'SIGI', 'NPCT', 'FRST', 'BN', 'VKTX', 'ESRT', 'CVLG', 'SLQT', 'TREE', 'STBA', 'ARGX', 'AB', 'SKX', 'CCB', 'UCTT', 'STEL', 'MSDL', 'TKC', 'RCS', 'CNSL', 'SWI', 'PODD', 'WD', 'BC', 'HOG', 'VVI', 'GRC', 'PGP', 'MFIC', 'HNI', 'ENVA', 'CHT', 'FEMY', 'FCO', 'INST', 'INFN', 'CCIF', 'MIO', 'AMED', 'CHK', 'RES', 'BALY', 'BMO', 'BUSE', 'MEDP', 'AEHL', 'AINC', 'CIK', 'HON', 'PCQ', 'NMZ', 'SBXC', 'ASB', 'FREE', 'ACP', 'BTO', 'ADNT', 'SLN', 'PGRU', 'ZVRA', 'DCF', 'CET', 'CMBT', 'ALIM', 'ATYR', 'HTBK', 'ALGM', 'COLM', 'HMN', 'TFII', 'TNDM', 'SDPI', 'IMNN', 'GGT', 'FBNC', 'INDV', 'HP', 'B', 'DMLP', 'TELL', 'RCAT', 'NCNC', 'NEWT', 'MGRC', 'GNTX', 'CRI', 'WSBC', 'UVE', 'EAF', 'XPRO', 'QTTB', 'LHX', 'BTE', 'CNTM', 'BFRG', 'IRTC', 'BIIB', 'OLN', 'AMTB', 'LPLA', 'ADN', 'OPY', 'KDLY', 'MXL', 'BAH', 'SKYW', 'URG', 'APPF', 'TARS', 'ZYXI', 'FBIO', 'BJRI', 'SAIA', 'ABVE', 'PMN', 'DXCM']
    Error fetching news for SBXC: 'NoneType' object has no attribute 'find_all'
    Error fetching news for CMBT: 'NoneType' object has no attribute 'find_all'


    We strongly recommend passing in an `attention_mask` since your input_ids may be padded. See https://huggingface.co/docs/transformers/troubleshooting#incorrect-output-when-padding-tokens-arent-masked.


    CSV file 'news_2024-07-27_to_2024-07-28.csv' has been created.


**Error Explanation** <br>
Error fetching news for [ticker]: 'NoneType' object has no attribute 'find_all' : this error indicates that there are no news released for the following ticker on finviz.

#Comparative Analysis


**Methodology:** <br>

1.   FinVader:<br>
uses a lexicon-based approach where it relies on predefined dictionaries of positive and negative words to calculate sentiment scores. incorporates specialized financial lexicons such as SentiBignomics and Henry's Financial Dictionary to better understand the financial context.

2.   NLTK's Sentiment Intensity Analyzer: <br>
lexicon and rule-based sentiment analysis tool that is general-purpose but effective in many contexts.

3.   FinBERT: <br>
 based on the BERT architecture, which is a deep learning model pre-trained on a large corpus of text and fine-tuned on financial texts. uses attention mechanisms to understand the context and relationships between words in a sentence.

**Sentiment Score Evaluation:** <br>

1.   FinVader:<br>
computed as a composite score that considers the presence and intensity of words in the text. provides a compound score which is a normalized, weighted composite score ranging from -1 to 1.

2.   NLTK's Sentiment Intensity Analyzer: <br>
includes rules that consider word order, degree modifiers, and punctuation to improve accuracy. range is also from -1 to 1.

3.   FinBERT: <br>
provides probabilities for each sentiment class (positive, neutral, negative) and a sentiment score is derived based on these probabilities.


**Effectiveness:** <br>

1.   FinVader:<br>
strength: effective in financial contexts due to its use of financial-specific lexicons. <br> weakness: performance can be limited by the quality and comprehensiveness of the lexicons

2.   NLTK's Sentiment Intensity Analyzer: <br>
strength:  performs well with texts that include slang, emojis, and varying punctuation. It's widely used due to its simplicity and effectiveness across different domains. <br> weakness: lack the depth of financial-specific sentiment understanding

3.   FinBERT: <br>
strength: excels in understanding context and complex sentence structures, making it highly effective for financial news sentiment analysis. <br> weakness: requires significant computational resources for training and large amounts of  data for fine-tuning to achieve optimal performance.

#Visualization

**Box Plot** <br>
The box plots aim to illustrate how sentiment scores derived from financial news articles are distributed across four distinct categories: 'Very Negative', 'Negative', 'Positive', and 'Very Positive'. These categories are created by dividing sentiment scores into bins using pd.cut, which assigns each score a label based on its value. By categorizing and visualizing these scores, the box plots provide a clear, comparative view of how each sentiment analysis model interprets the sentiment of financial news. This visualization helps in identifying biases or differences in sentiment classification among the models.


```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


#Convert specified columns in the DataFrame to numeric, handling non-numeric values.
def clean_and_convert_to_numeric(df, columns):
    for column in columns:
        if column == 'Change':
            # Remove percentage signs and convert to numeric
            df[column] = df[column].replace('%', '', regex=True)  # Remove percentage signs
        # Convert to numeric
        df[column] = pd.to_numeric(df[column], errors='coerce')
    return df

def plot_box_plots(news_df):
    # Create bins for sentiment scores to fall into four different categories.
    bins = [-np.inf, -0.5, 0, 0.5, np.inf]
    labels = ['Very Negative', 'Negative', 'Positive', 'Very Positive']
    news_df['SS_Finvader_binned'] = pd.cut(news_df['SS_Finvader'], bins=bins, labels=labels)
    news_df['SS_NLTK_binned'] = pd.cut(news_df['SS_NLTK'], bins=bins, labels=labels)
    news_df['SS_FinRoberta_binned'] = pd.cut(news_df['SS_FinRoberta'], bins=bins, labels=labels)

    # Plot box plots
    plt.figure(figsize=(14, 14))

    plt.subplot(3, 1, 1)
    sns.boxplot(x='SS_Finvader_binned', y='Change', data=news_df)
    plt.title('Distribution of Change in Stock Price for SS_Finvader Bins')
    plt.xlabel('SS_Finvader')
    plt.ylabel('Change')

    plt.subplot(3, 1, 2)
    sns.boxplot(x='SS_NLTK_binned', y='Change', data=news_df)
    plt.title('Distribution of Change in Stock Price for SS_NLTK Bins')
    plt.xlabel('SS_NLTK')
    plt.ylabel('Change')

    plt.subplot(3, 1, 3)
    sns.boxplot(x='SS_FinRoberta_binned', y='Change', data=news_df)
    plt.title('Distribution of Change in Stock Price for SS_FinRoberta Bins')
    plt.xlabel('SS_FinRoberta')
    plt.ylabel('Change')

    plt.tight_layout()
    plt.show()

def load_and_display_box_plots(csv_filename):
    # Load the data
    news_df = pd.read_csv(csv_filename)

    # Clean and convert relevant columns to numeric
    columns_to_clean = ['SS_Finvader', 'SS_NLTK', 'SS_FinRoberta', 'Change']
    news_df = clean_and_convert_to_numeric(news_df, columns_to_clean)

    # Drop rows with NaN values after conversion
    news_df = news_df.dropna(subset=columns_to_clean)

    # Display the box plots
    plot_box_plots(news_df)

# Update with the correct filename based on the date range used in the main script in the cell above.
csv_filename = 'news_2024-07-27_to_2024-07-28.csv'
load_and_display_box_plots(csv_filename)

```


    
![png](Finanacial_News_Sentiment_Analysis_files/Finanacial_News_Sentiment_Analysis_18_0.png)
    


**Finding** <br>
The analysis of the box plots reveals significant differences in how each model distributes sentiment scores. Finvader predominantly classifies news articles as 'Very Positive', indicating a potential bias towards positive sentiment. Similarly, NLTK shows a strong skew towards 'Very Positive' categories, suggesting that this general-purpose sentiment analyzer may not fully capture the nuances of negative sentiment in financial contexts. In contrast, FinBERT exhibits a more balanced distribution across all categories, including substantial representations in 'Very Negative' 'Negative', and 'Positive'. This suggests that FinBERT, with its transformer-based architecture and financial text pre-training, is more likely to capture a wider range of sentiments.

#Evaluation Metric

**Functions** <br>

**The clean_and_convert_to_numeric:** <br>  removes percentage signs from the 'Change' column and converts the specified columns to numeric values. It also categorizes stock price changes into 'Up' or 'Down' based on whether the 'Change' value is positive or negative.

**categorize_sentiment:** <br>
assigns 'Up' or 'Down' labels to sentiment scores based on whether they are above or below zero.

**create_confusion_matrix:** <br>
generates a confusion matrix for each sentiment model by comparing the predicted sentiment categories to the actual stock price change categories.




**Confusion Matrix** <br>
The confusion matrix compares the predicted sentiment categories  against the actual stock price change categories. It is created from sklearn.metrics, which counts the number of true positives (model correctly predicts an "Up" movement in stock prices), true negatives (model correctly predicts a "Down" movement in stock prices), false positives (model incorrectly predicts an "Up" movement), and false negatives (model incorrectly predicts "Down" movement).



**Evaluation Metrics** <br>
Accuracy is the proportion of the total number of correct predictions (both 'Up' and 'Down') to the total number of predictions.
Precision for 'Up' is calculated as the number of true positives divided by the sum of true positives and false positives.
Recall for 'Up' is calculated as the number of true positives divided by the sum of true positives and false negatives.



```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Define the data cleaning function
def clean_and_convert_to_numeric(df, columns):
    for column in columns:
        if column == 'Change':
            # Remove percentage signs and convert to numeric
            df[column] = df[column].replace('%', '', regex=True)  # Remove percentage signs
        # Convert to numeric
        df[column] = pd.to_numeric(df[column], errors='coerce')
    # Categorize the stock price changes
    df['Price_Change_Category'] = np.where(df['Change'] > 0, 'Up', 'Down')
    return df

# Load the data
csv_filename = 'news_2024-07-27_to_2024-07-28.csv'
news_df = pd.read_csv(csv_filename)

# Clean and convert relevant columns to numeric
columns_to_clean = ['SS_Finvader', 'SS_NLTK', 'SS_FinRoberta', 'Change']
news_df = clean_and_convert_to_numeric(news_df, columns_to_clean)

# Drop rows with NaN values after conversion
news_df = news_df.dropna(subset=columns_to_clean)

# Categorize sentiment scores as Up or Down based on whether they are above or below zero
def categorize_sentiment(df, sentiment_col):
    df[f'{sentiment_col}_category'] = np.where(df[sentiment_col] > 0, 'Up', 'Down')
    return df

news_df = categorize_sentiment(news_df, 'SS_Finvader')
news_df = categorize_sentiment(news_df, 'SS_NLTK')
news_df = categorize_sentiment(news_df, 'SS_FinRoberta')

# Define a function to create a confusion matrix for each sentiment model
def create_confusion_matrix(df, sentiment_col):
    cm = confusion_matrix(df['Price_Change_Category'], df[f'{sentiment_col}_category'], labels=['Up', 'Down'])
    return cm

# Define a function to calculate evaluation metrics from the confusion matrix
def calculate_metrics(df, sentiment_col):
    y_true = df['Price_Change_Category']
    y_pred = df[f'{sentiment_col}_category']
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    accuracy = report['accuracy']
    precision = report['Up']['precision']
    recall = report['Up']['recall']
    return accuracy, precision, recall

# Define a function to plot a confusion matrix heatmap
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Up', 'Predicted Down'], yticklabels=['Actual Up', 'Actual Down'])
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Evaluate and plot confusion matrices for each sentiment score
for sentiment_col in ['SS_Finvader', 'SS_NLTK', 'SS_FinRoberta']:
    cm = create_confusion_matrix(news_df, sentiment_col)
    accuracy, precision, recall = calculate_metrics(news_df, sentiment_col)
    print(f"{sentiment_col} - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")
    plot_confusion_matrix(cm, f'Confusion Matrix for {sentiment_col}')

```

    SS_Finvader - Accuracy: 0.73, Precision: 0.73, Recall: 1.00



    
![png](Finanacial_News_Sentiment_Analysis_files/Finanacial_News_Sentiment_Analysis_23_1.png)
    


    SS_NLTK - Accuracy: 0.73, Precision: 0.73, Recall: 1.00



    
![png](Finanacial_News_Sentiment_Analysis_files/Finanacial_News_Sentiment_Analysis_23_3.png)
    


    SS_FinRoberta - Accuracy: 0.45, Precision: 0.75, Recall: 0.38



    
![png](Finanacial_News_Sentiment_Analysis_files/Finanacial_News_Sentiment_Analysis_23_5.png)
    


**Findings** <br>

Compared to the other two models, FinBERT has lower accuracy, indicating less reliability in predicting stock movements. It only identified 38% of the actual "Up" movements, suggesting that while it provides a broader range of sentiment analysis, it may not be as effective in translating those sentiments into precise stock movement predictions. <br>

On the other hand, the Finvader and NLTK models demonstrated high effectiveness in predicting "Up" movements. This suggests they can be more reliable tools for identifying potential upward trends in stock prices. However, both models failed to accurately predict any "Down" movements, indicating a significant limitation. Their inability to capture negative sentiments means these models may not provide a comprehensive view of the market, potentially overlooking critical downside risks.


#Challenge & Limitation

**Challenges**


1.   **Fetching Stock Data Using API URL:** <br>
Initially, fetching stock data from the API was challenging due to unfamiliarity with the API's structure and parameters. By leveraging online resources, I was able to successfully retrierve the data needed for analysis.

2.   **Hanlding News Released on "Today":** <br>
Finviz label news article as "Today" isntead of providing a specific date. This inconsistency posed a challenge during data preprocessing, as it required special handling to ensure accurate date parsing and alignment with stock price data. Implementing a dynamic solution to convert "Today" into the current date while maintaining the correct format for other dates was crucial.

3. **429 Client Error:** <br>
Encountering the 429 error, which indicates that the rate limit for API requests was exceeded, was another significant challenge. To overcome this, I implemented a retry mechanism with exponential backoff. This strategy involved waiting progressively longer intervals between retries, ensuring compliance with the API's rate limits while avoiding disruptions in data collection.

4. **Issue in FinBERT Model Implementation:** <br>
The error occurs since the FinBERT model has a maximum token limit of 512 tokens, and some of the processed content exceeds this limit. To handle this, I had to  truncate or split the content into smaller chunks before processing them with the FinBERT model.

5. **Computing Evaluation Metrics:** <br>
Had to determine sentiment score ranges for categorizing sentiment which was necessary step for evaluating the model's performance. Also required several modifications to accurately compute evaluation metrics.

**Limitations**

1.   **Data Diversity:** <br>
The quality and coverage of the financial news data can vary, impacting the robustness of sentiment analysis. News articles might have different lengths, tones, and reporting styles, which can influence sentiment scores. Additionally, some relevant news might be missed due to limitations in the news sources or the date range selected for the study.

2.   **Market Influences Beyond News Sentiment:** <br>
Stock price movements are influenced by numerous factors beyond news sentiment, such as macroeconomic indicators, market trends, investor behavior, and company-specific events. Isolating the impact of news sentiment on stock prices is inherently challenging, and I believe the study's findings might be influenced by these external variables.

3.   **Computational Constraints:** <br>
Processing large volumes of text data and applying complex sentiment analysis models require significant computational resources. This project was constrained by the available computational power, potentially limiting the scale of data that could be processed and analyzed efficiently.

4.   **Generalizability of Findings:** <br>
The findings from this study are based on a specific dataset of news articles and stock tickers within a defined time frame. The generalizability of the results to other time periods, market conditions, or financial instruments might be limited. Further research with enriched datasets and extended time frames would be inevitable to validate and generalize the conclusions.








#Future Work

1.   Model Improvement: <br>
I could explore and develop more sophisticated sentiment analysis models that improve their ability to understand and accurately capture contextual nuances unique to the financial domain. Leveraging advanced NLP techniques could also further enhance the result.

2.   Data Sources Expansion: <br>
Future research could include a wider range of news sources. This would involve integrating news from various financial news platforms other than finviz, social media, and forums, offering a broader perspective on market sentiment.

3.   Evaluation Metric Enhancement: <br>
Implementing cross-validation and backtesting techniques to evaluate the predictive performance of sentiment  models would ensure their robustness and reliability in different scenarios.

4.   Addressing Other Limitations: <br>
Further research would focus on identifying biases introduced by the training data and developing methods to mitigate them, ensuring fair sentiment analysis. Also, enhanced data quality by implementing more sophisticated preprocessing techniques and handling missing data more effectively would be beneficial.

