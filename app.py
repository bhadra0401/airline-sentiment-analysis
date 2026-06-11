import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import re
from nltk.corpus import stopwords
from wordcloud import WordCloud

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="Airline Sentiment Analyzer",
    page_icon="",
    layout="wide"
)

# ======================================================
# LOAD MODEL
# ======================================================

model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ======================================================
# STOPWORDS
# ======================================================

try:
    stop_words = set(stopwords.words("english"))
except:
    import nltk
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

# ======================================================
# TEXT CLEANING
# ======================================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# ======================================================
# LOAD DATASET
# ======================================================

try:
    df = pd.read_csv("Tweets.csv")
except:
    df = pd.DataFrame()

# ======================================================
# SIDEBAR
# ======================================================

st.sidebar.title(" Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "Sentiment Analyzer",
        "Dashboard",
        "Bulk Prediction",
        "About"
    ]
)

# ======================================================
# SENTIMENT ANALYZER
# ======================================================

if page == "Sentiment Analyzer":

    st.title(" Airline Review Sentiment Analyzer")

    st.markdown(
        """
        Analyze airline customer reviews using
        Machine Learning and NLP.
        """
    )

    col1, col2 = st.columns([2, 1])

    with col1:

        review = st.text_area(
            "Enter Airline Review",
            height=200
        )

        analyze = st.button("Analyze Sentiment")

    with col2:

        st.info(
            """
            Examples:

             Great flight experience

             Flight delayed for 6 hours

             Flight was okay
            """
        )

    if analyze and review:

        cleaned = clean_text(review)

        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)[0]

        probabilities = model.predict_proba(vector)

        confidence = probabilities.max() * 100

        st.subheader("Prediction Result")

        if prediction.lower() == "positive":
            st.success(" Positive Sentiment")

        elif prediction.lower() == "negative":
            st.error(" Negative Sentiment")

        else:
            st.warning(" Neutral Sentiment")

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

# ======================================================
# DASHBOARD
# ======================================================

elif page == "Dashboard":

    st.title(" Sentiment Dashboard")

    if df.empty:

        st.error("Tweets.csv not found")

    else:

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Reviews",
            len(df)
        )

        col2.metric(
            "Airlines",
            df["airline"].nunique()
        )

        col3.metric(
            "Sentiment Classes",
            df["airline_sentiment"].nunique()
        )

        st.markdown("---")

        st.subheader("Overall Sentiment Distribution")

        fig, ax = plt.subplots(figsize=(8, 4))

        sns.countplot(
            x="airline_sentiment",
            data=df,
            ax=ax
        )

        st.pyplot(fig)

        st.markdown("---")

        st.subheader("Airline-wise Sentiment Analysis")

        fig, ax = plt.subplots(figsize=(12, 5))

        sns.countplot(
            x="airline",
            hue="airline_sentiment",
            data=df,
            ax=ax
        )

        plt.xticks(rotation=45)

        st.pyplot(fig)

        st.markdown("---")

        st.subheader("Word Cloud")

        if "text" in df.columns:

            text = " ".join(
                df["text"].astype(str)
            )

            wordcloud = WordCloud(
                width=1000,
                height=500,
                background_color="white"
            ).generate(text)

            fig, ax = plt.subplots(
                figsize=(12, 6)
            )

            ax.imshow(wordcloud)

            ax.axis("off")

            st.pyplot(fig)

# ======================================================
# BULK PREDICTION
# ======================================================

elif page == "Bulk Prediction":

    st.title(" Bulk Sentiment Prediction")

    st.markdown(
        """
        Upload a CSV file containing a column
        named **text**
        """
    )

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded_file:

        bulk_df = pd.read_csv(uploaded_file)

        if "text" not in bulk_df.columns:

            st.error(
                "CSV must contain a column named 'text'"
            )

        else:

            bulk_df["clean_text"] = (
                bulk_df["text"]
                .astype(str)
                .apply(clean_text)
            )

            vectors = vectorizer.transform(
                bulk_df["clean_text"]
            )

            bulk_df["prediction"] = (
                model.predict(vectors)
            )

            st.success(
                "Prediction Completed!"
            )

            st.dataframe(bulk_df)

            csv = bulk_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇ Download Predictions",
                csv,
                "predictions.csv",
                "text/csv"
            )

# ======================================================
# ABOUT
# ======================================================

elif page == "About":

    st.title(" About Project")

    st.markdown(
        """
        ## Airline Sentiment Analysis

        This project uses:

        - Natural Language Processing (NLP)
        - TF-IDF Vectorization
        - Logistic Regression
        - Streamlit Dashboard

        ### Model Performance

        - Logistic Regression: 79.85%
        - Random Forest: 77.29%
        - Naive Bayes: 73.77%

        ### Features

         Real-time Sentiment Prediction

         Dashboard Analytics

         Word Cloud Visualization

         Bulk CSV Prediction

         Download Prediction Results

        ### Developed By

        Naga Veera Bhadra Kumar Akkala
        """
    )