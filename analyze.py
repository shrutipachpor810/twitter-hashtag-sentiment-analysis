# local imports
from time import sleep

# module imports
import re
import sys
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from wordcloud import WordCloud  # 🔥 NEW

# -------------------------------
# INIT
# -------------------------------
lag = 0.03

sentiment_model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment"
)

label_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

def writer(text):
    for char in text:
        sleep(lag)
        sys.stdout.write(char)
        sys.stdout.flush()

# -------------------------------
# CLEANING
# -------------------------------
def clean_tweet(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s#]", "", text)
    return text.strip()

# -------------------------------
# WORDCLOUD FUNCTION 🔥
# -------------------------------
def generate_wordcloud(text_series, hashtag):
    text = " ".join(text_series)

    if text.strip() == "":
        return

    wc = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure()
    plt.imshow(wc)
    plt.axis("off")
    plt.title(f"WordCloud for #{hashtag}")
    plt.show()

# -------------------------------
# SENTIMENT + POLARITY
# -------------------------------
def analyze_text(text):
    try:
        result = sentiment_model(text[:512])[0]

        label = label_map[result['label']]
        score = result['score']

        if label == "Neutral":
            polarity = 0
        elif label == "Positive":
            polarity = score
        else:
            polarity = -score

        return label, polarity, score

    except:
        return "Neutral", 0, 0

# -------------------------------
# LOAD + FILTER
# -------------------------------
def load_and_filter_tweets(hash):

    df = pd.read_csv("full-corpus.csv")

    text_col = None
    for col in df.columns:
        if 'text' in col.lower() or 'tweet' in col.lower():
            text_col = col

    if text_col is None:
        print("❌ Text column not found")
        return pd.DataFrame()

    df = df[[text_col]].dropna()
    df.columns = ['text']

    keyword = hash.lower().replace("#", "")

    filtered_df = df[
        df['text'].str.lower().str.contains(f"#{keyword}") |
        df['text'].str.lower().str.contains(rf"\b{keyword}\b", regex=True)
    ]

    return filtered_df.sample(min(len(filtered_df), 300)) if not filtered_df.empty else pd.DataFrame()

# -------------------------------
# MAIN ANALYSIS
# -------------------------------
def analyze(hash):

    writer(f'\nAnalyzing #{hash} sentiment using Twitter RoBERTa...\n')

    df = load_and_filter_tweets(hash)
    keyword = hash.replace("#", "")

    # --------------------------------
    # CASE 1: DATASET EXISTS
    # --------------------------------
    if not df.empty:

        df['clean'] = df['text'].apply(clean_tweet)
        df = df[df['clean'] != ""]

        results = df['clean'].apply(analyze_text)
        df[['Sentiment', 'Polarity', 'Confidence']] = pd.DataFrame(results.tolist(), index=df.index)

        pos = len(df[df['Sentiment'] == 'Positive'])
        neg = len(df[df['Sentiment'] == 'Negative'])
        neu = len(df[df['Sentiment'] == 'Neutral'])

        total = len(df)

        pos_p = round((pos / total) * 100, 2)
        neg_p = round((neg / total) * 100, 2)
        neu_p = round((neu / total) * 100, 2)

        avg_polarity = round(df['Polarity'].mean(), 3)
        trend_score = round((pos - neg) / total, 3)

        print(f"\nHashtag: #{hash}")
        print(f"Positive: {pos_p}% | Negative: {neg_p}% | Neutral: {neu_p}%")
        print(f"Total Tweets: {total}")
        print(f"Average Polarity: {avg_polarity}")
        print(f"Trend Score: {trend_score}")

        print("\n🔥 Top Positive Tweets:")
        print(df.sort_values(by='Polarity', ascending=False)['text'].head(2).to_string(index=False))

        print("\n💀 Top Negative Tweets:")
        print(df.sort_values(by='Polarity')['text'].head(2).to_string(index=False))

        # -------------------------------
        # VISUALS
        # -------------------------------

        # Bar chart
        plt.figure()
        plt.bar(['Positive', 'Negative', 'Neutral'], [pos, neg, neu])
        plt.title(f"Sentiment Distribution for #{hash}")
        plt.show()

        # Polarity distribution
        plt.figure()
        plt.hist(df['Polarity'], bins=20)
        plt.title("Polarity Distribution")
        plt.xlabel("Polarity (-1 to +1)")
        plt.ylabel("Frequency")
        plt.show()

        # 🔥 WORDCLOUD
        generate_wordcloud(df['clean'], hash)

    # --------------------------------
    # CASE 2: NO DATA
    # --------------------------------
    else:
        print("⚠️ No dataset found → using direct RoBERTa analysis\n")

        text = f"This is about {keyword}"
        sentiment, polarity, confidence = analyze_text(text)

        print(f"Hashtag: #{hash}")
        print(f"Sentiment: {sentiment}")
        print(f"Polarity: {round(polarity, 3)}")
        print(f"Confidence: {round(confidence, 3)}")

        plt.figure()
        plt.bar(['Positive', 'Negative', 'Neutral'],
                [1 if sentiment=="Positive" else 0,
                 1 if sentiment=="Negative" else 0,
                 1 if sentiment=="Neutral" else 0])
        plt.title(f"Prediction for #{hash}")
        plt.show()

# -------------------------------
# CLI
# -------------------------------
if __name__ == "__main__":

    while True:
        print("\nEnter #hashtag | -1 to exit\n")
        tag = input("> ")

        if tag == "-1":
            break

        analyze(tag)