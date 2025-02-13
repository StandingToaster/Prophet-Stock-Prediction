import requests
from transformers import pipeline
from datetime import datetime
import openai
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Initialize OpenAI client
client = openai.OpenAI()

def get_financial_news(ticker):
    """Fetches stock-related news with formatted timestamps from NewsAPI.org."""
    
    url = f"https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&language=en&apiKey={NEWS_API_KEY}"

    response = requests.get(url)

    if response.status_code != 200:
        print(f"⚠ Error fetching news: {response.status_code} - {response.text}")
        return []

    news_data = response.json()

    headlines = []
    for article in news_data.get("articles", []):
        title = article["title"]
        url = article["url"]
        raw_timestamp = article["publishedAt"]
        formatted_timestamp = datetime.strptime(raw_timestamp, "%Y-%m-%dT%H:%M:%SZ").strftime("%b %d, %Y %I:%M %p")
        headlines.append((title, url, formatted_timestamp))

    return headlines[:3] if headlines else []


def analyze_sentiment(news_articles):
    """Uses GPT-3.5 to analyze sentiment of financial news headlines."""
    results = []

    for title, url, timestamp in news_articles:
        prompt = f"""
        Analyze the sentiment of this financial news headline:
        Headline: "{title}"
        
        Classify the sentiment as: Positive, Neutral, or Negative.
        Explain why you classified it this way in a short sentence.
        """

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "You are a financial sentiment analysis expert."},
                          {"role": "user", "content": prompt}],
                max_tokens=50
            )
            sentiment_analysis = response.choices[0].message.content

            # Extract sentiment classification
            sentiment = "neutral"  # Default
            if "Positive" in sentiment_analysis:
                sentiment = "positive"
            elif "Negative" in sentiment_analysis:
                sentiment = "negative"

            results.append((title, url, timestamp, sentiment, sentiment_analysis))

        except Exception as e:
            results.append((title, url, timestamp, "error", str(e)))

    return results