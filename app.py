import streamlit as st;
import preprocessor, helper
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from matplotlib import rcParams

# Use a font that supports emojis
plt.rcParams['font.family'] = 'Segoe UI Emoji'
st.set_page_config(layout="wide")

# Inject custom CSS
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"]{
        
        background: linear-gradient(to bottom, #25D366, #128C7E); /* WhatsApp green gradient */
        color: crimson;    
       
        max-width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] {
        background: linear-gradient(to bottom, #25D366, #128C7E); /* WhatsApp green gradient */
        min-width: 0px;
        max-width: 0px;
    }

    
    </style>
    """,
    unsafe_allow_html=True
)



st.sidebar.title("Whatsapp Chat Analyzer")
st.markdown(
    """
    <style>
    /* Change the uploader box background */
    
    [data-testid="stFileUploader"] {
        background-color: black;  /* Light pink */
        border: 2px dashed black;
        padding: 20px;
        border-radius: 10px;
        color: white;
    }

    /* Change the upload text */
    [data-testid="stFileUploader"] section div div span {
        color: crimson;
        font-weight: bold;
        font-size: 16px;
    }
    
    [data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1633675254245-efd890d087b8?q=80&w=1074&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
    background-size: cover;
    background-position: right;
    background-repeat: repeat;
    background-attachment: fixed;
}
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    '<p style="color: crimson; font-size:16px; font-weight:bold;">📂 Upload your file below</p>',
    unsafe_allow_html=True
)

# File uploader without label
uploaded_file = st.file_uploader("", key="uploader")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    user_list = df['user'].unique().tolist()
    user_list.remove('group_notofication')
    user_list.sort()
    user_list.insert(0, "🌍 Overall")  # Add overall option

    selected_user = st.sidebar.selectbox(
        "👤 Show analysis for",
        user_list,
        index=0
    )

    if st.sidebar.button("Show Analysis"):
        num_messages, words, num_media_messages, num_links, num_deleted_message = helper.fetch_stats(selected_user, df)
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown("<h4 style='color:crimson;'>📩 Messages</h4>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='color:black;'>{num_messages}</h2>", unsafe_allow_html=True)

        with col2:
            st.markdown("<h4 style='color:orange;'>📝 Words</h4>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='color:black;'>{words}</h2>", unsafe_allow_html=True)

        with col3:
            st.markdown("<h4 style='color:purple;'>❌ Deleted</h4>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='color:black;'>{num_deleted_message}</h2>", unsafe_allow_html=True)
        with col4:
            st.markdown("<h4 style='color:blue;'>🖼️ Media</h4>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='color:black;'>{num_media_messages}</h2>", unsafe_allow_html=True)
        with col5:
            st.markdown("<h4 style='color:blue;'>🖼🔗 Links</h4>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='color:black;'>{num_links}</h2>", unsafe_allow_html=True)
        analyzer = SentimentIntensityAnalyzer()

        # Filter by selected user
        if selected_user != "🌍 Overall":
            temp_df = df[df['user'] == selected_user]
        else:
            temp_df = df.copy()

        sentiments = {"Positive": 0, "Negative": 0, "Neutral": 0}
        scores = []

        for message in temp_df['message']:
            if message not in ["<Media omitted>", "This message was deleted"]:
                vs = analyzer.polarity_scores(str(message))
                scores.append(vs['compound'])
                if vs['compound'] >= 0.05:
                    sentiments["Positive"] += 1
                elif vs['compound'] <= -0.05:
                    sentiments["Negative"] += 1
                else:
                    sentiments["Neutral"] += 1

        sentiment_df = pd.DataFrame(sentiments.items(), columns=["Sentiment", "Count"])

        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(sentiment_df)
        with col2:
            fig, ax = plt.subplots()
            cmap = cm.get_cmap('Set2')
            colors = cmap(np.linspace(0, 1, len(sentiment_df)))
            ax.pie(sentiment_df["Count"], labels=sentiment_df["Sentiment"], autopct="%0.1f%%", colors=colors)
            st.pyplot(fig)
        if selected_user == '🌍 Overall':
            x, new_df = helper.fetch_most_bysy_users(df)
            st.markdown(
                "<h3 style='color:crimson; text-align:center; font-weight:bold;'>🔥 Most Busy Users 🔥</h3>",
                unsafe_allow_html=True
            )
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)
            with col1:
                ax.bar(x.index, x.values, color="#FFA500", edgecolor="black", linewidth=1.2)
                plt.xticks(rotation='vertical')
                ax.set_ylabel("Messages", fontsize=12)
                ax.set_xlabel("Users", fontsize=12)
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)
        df_wc = helper.create_wordcloud(selected_user, df)
        st.markdown(
            "<h3 style='color:crimson; text-align:center; font-weight:bold;'>Word Cloud</h3>",
            unsafe_allow_html=True
        )
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        most_common_df = helper.most_common_words(selected_user, df)
        fig, ax = plt.subplots()
        st.markdown(
            "<h3 style='color:crimson; text-align:center; font-weight:bold;'>Most Common Words</h3>",
            unsafe_allow_html=True
        )
        ax.barh(most_common_df[0], most_common_df[1], color="#FFA500", edgecolor="black", linewidth=1.2)
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        emoji_df = helper.emoji_helper(selected_user, df)
        st.markdown(
            "<h3 style='color:crimson; text-align:center; font-weight:bold;'>Emoji Analysis</h3>",
            unsafe_allow_html=True
        )
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(emoji_df)
        with col2:
            cmap = cm.get_cmap('tab20')  # 'tab20', 'Set3', 'Paired', etc.

            colors = cmap(np.linspace(0, 1, len(emoji_df)))  # Generate colors
            fig, ax = plt.subplots()
            fig.patch.set_facecolor('lightgreen')
            ax.pie(emoji_df[1], labels=emoji_df[0], autopct="%0.2f%%", colors=colors)
            st.pyplot(fig)
        timeline = helper.monthly_timeline(selected_user, df)
        st.markdown(
            "<h3 style='color:crimson; text-align:center; font-weight:bold;'>Monthly Timeline</h3>",
            unsafe_allow_html=True
        )
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color = "orange")
        plt.xticks(rotation="vertical")
        st.pyplot(fig)

        daily_timeline = helper.daily_timeline(selected_user, df)
        st.markdown(
            "<h3 style='color:crimson; text-align:center; font-weight:bold;'>Daily Timeline</h3>",
            unsafe_allow_html=True
        )
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color="orange")
        plt.xticks(rotation="vertical")
        st.pyplot(fig)

        st.markdown(
            "<h3 style='color:crimson; text-align:center; font-weight:bold;'>Activity Map</h3>",
            unsafe_allow_html=True
        )
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                "<h3 style='color:crimson; text-align:center; font-weight:bold;'>Most busy day</h3>",
                unsafe_allow_html=True
            )
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color="orange")
            plt.xticks(rotation="vertical")
            st.pyplot(fig)
        with col2:
            st.markdown(
                "<h3 style='color:crimson; text-align:center; font-weight:bold;'>Most busy month</h3>",
                unsafe_allow_html=True
            )
            busy_month = helper.monthly_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color="orange")
            plt.xticks(rotation="vertical")
            st.pyplot(fig)

