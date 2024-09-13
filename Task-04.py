import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load your dataset Replace this path with the path to your dataset
df = pd.read_csv('C:/Users/ASUS/Documents/GitHub/PRODIGY-INFOTECH-Task-04/twitter_training.csv')

print(df.columns)

print(df.head())

df = df.dropna(subset=['Text']) 
df['Text'] = df['Text'].astype(str) 

analyzer = SentimentIntensityAnalyzer()

df['VADER_Sentiment'] = df['Text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

print(df[['Sentiment', 'VADER_Sentiment', 'Text']])

plt.figure(figsize=(10,6))
plt.hist(df['VADER_Sentiment'], bins=20, color='blue', alpha=0.7)
plt.title('Distribution of VADER Sentiment Scores')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
