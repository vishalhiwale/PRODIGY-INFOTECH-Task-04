import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud

# Load the dataset
data = pd.read_csv('C:/Users/uu/Documents/GitHub/PRODIGY-INFOTECH-Task-04/twitter_validation.csv')
# print("Column names in the dataset:")
# print(data.columns.tolist())
# Display basic information
print("Dataset Info:")
print(data.info())
print("\nColumn names in the dataset:")
print(data.columns)
print("\nSentiment Distribution:")
print(data['sentiment'].value_counts())

# Aggregate and visualize sentiment counts
sentiment_column = 'sentiment_label'  # Adjust to your actual sentiment column name
if sentiment_column in data.columns:
    sentiment_counts = data[sentiment_column].value_counts()
    plt.figure(figsize=(10, 6))
    sentiment_counts.plot(kind='bar', color=['green', 'gray', 'red'])
    plt.title('Distribution of Sentiments')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Tweets')
    plt.xticks(rotation=45)
    plt.show()
else:
    print(f"Column '{sentiment_column}' does not exist in the dataset.")

# Sentiment distribution by entity, if 'entity' column exists
entity_column = 'entity'  # Adjust to your actual entity column name
if entity_column in data.columns:
    entity_sentiments = data.groupby([entity_column, sentiment_column]).size().unstack().fillna(0)
    entity_sentiments.plot(kind='bar', stacked=True, figsize=(14, 8))
    plt.title('Sentiment Distribution by Entity')
    plt.xlabel('Entity')
    plt.ylabel('Number of Tweets')
    plt.xticks(rotation=45)
    plt.legend(title='Sentiment')
    plt.show()
else:
    print(f"Column '{entity_column}' does not exist in the dataset.")

# Convert timestamp to datetime if 'timestamp' column exists
timestamp_column = 'timestamp'  # Adjust to your actual timestamp column name
if timestamp_column in data.columns:
    data[timestamp_column] = pd.to_datetime(data[timestamp_column])
    data['date'] = data[timestamp_column].dt.date
    daily_sentiments = data.groupby(['date', sentiment_column]).size().unstack().fillna(0)
    daily_sentiments.plot(figsize=(14, 8))
    plt.title('Daily Sentiment Trends')
    plt.xlabel('Date')
    plt.ylabel('Number of Tweets')
    plt.legend(title='Sentiment')
    plt.show()
else:
    print(f"Column '{timestamp_column}' does not exist in the dataset.")

# Apply sentiment analysis to text and generate sentiment scores
text_column = 'text'  # Adjust to your actual text column name
if text_column in data.columns:
    data['sentiment_score'] = data[text_column].apply(lambda text: TextBlob(text).sentiment.polarity)

    # Plot sentiment score distribution
    plt.figure(figsize=(10, 6))
    plt.hist(data['sentiment_score'], bins=30, color='blue', alpha=0.7)
    plt.title('Sentiment Score Distribution')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Number of Tweets')
    plt.show()
else:
    print(f"Column '{text_column}' does not exist in the dataset.")

# Generate word clouds for positive and negative sentiments
if sentiment_column in data.columns and text_column in data.columns:
    positive_tweets = data[data[sentiment_column] == 'Positive'][text_column]
    negative_tweets = data[data[sentiment_column] == 'Negative'][text_column]

    positive_text = ' '.join(tweet for tweet in positive_tweets)
    negative_text = ' '.join(tweet for tweet in negative_tweets)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(WordCloud(background_color='white').generate(positive_text))
    plt.title('Positive Sentiment Word Cloud')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(WordCloud(background_color='white', colormap='Reds').generate(negative_text))
    plt.title('Negative Sentiment Word Cloud')
    plt.axis('off')

    plt.show()
else:
    print(f"Columns '{sentiment_column}' or '{text_column}' do not exist in the dataset.")
