# ================================
# FULL DATA PREPROCESSING SCRIPT
# ================================

import re
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -------------------------------
# Download required resources
# -------------------------------
nltk.download('stopwords')
nltk.download('wordnet')

# -------------------------------
# Initialize tools
# -------------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# -------------------------------
# CLEANING FUNCTION
# -------------------------------
def clean_tweet(text):
    if pd.isna(text):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # 3. Remove mentions (@user)
    text = re.sub(r"@\w+", "", text)

    # 4. Remove hashtag symbol (#word → word)
    text = re.sub(r"#", "", text)

    # 5. Remove numbers
    text = re.sub(r"\d+", "", text)

    # 6. Remove punctuation & special chars
    text = re.sub(r"[^\w\s]", "", text)

    # 7. Tokenize
    words = text.split()

    # 8. Remove stopwords + lemmatize
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    # 9. Join back
    return " ".join(words)

# -------------------------------
# LOAD DATASET
# -------------------------------
df = pd.read_csv("cleaned_dataset.csv")  # change path if needed

# 👉 IMPORTANT: change column name if different
TEXT_COLUMN = "TweetText"

# -------------------------------
# APPLY PREPROCESSING
# -------------------------------
df["clean_text"] = df[TEXT_COLUMN].apply(clean_tweet)

# -------------------------------
# REMOVE EMPTY / BAD ROWS
# -------------------------------
df = df[df["clean_text"] != ""]
df = df.drop_duplicates(subset="clean_text")

# Optional: remove very short tweets
df = df[df["clean_text"].str.len() > 5]

# -------------------------------
# SAVE CLEAN DATASET
# -------------------------------
df.to_csv("cleaned_dataset.csv", index=False)

# -------------------------------
# DONE
# -------------------------------
print("✅ Preprocessing complete!")
print(f"Original size: {len(pd.read_csv('full-corpus.csv'))}")
print(f"Cleaned size: {len(df)}")
print("Saved as: cleaned_dataset.csv")
