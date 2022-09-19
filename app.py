from ast import keyword
import streamlit as st
import numpy as np
from pandas import DataFrame
import seaborn as sns
import os
import json

import nltk
from keybert import KeyBERT
from flair.embeddings import TransformerDocumentEmbeddings
from rake_nltk import Rake
import yake
from yake.highlight import TextHighlighter

nltk.download('stopwords')
nltk.download('punkt')


st.set_page_config(
    page_title="DAEN 690 Team Concorde: NLP Algorithms",
    layout="wide"
)


def _max_width_():
    max_width_str = f"max-width: 1400px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )


# _max_width_()

c30, = st.columns(1)

with c30:
    st.title("DAEN 690 Team Concorde: NLP Algorithms")


logo1, logo2 = st.columns([3, 19])
with logo1:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/53/George_Mason_University_logo.svg/2560px-George_Mason_University_logo.svg.png", width=120)
with logo2:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c8/Seal_of_the_United_States_Federal_Aviation_Administration.svg/1200px-Seal_of_the_United_States_Federal_Aviation_Administration.svg.png", width=120)

st.markdown("")
st.markdown("## ✏️ Paste text")

with st.form(key="my_form"):

    ce, c1, ce, c2, c3 = st.columns([0.07, 1, 0.07, 5, 0.07])
    with c1:
        ModelType = st.radio(
            "Choose your model",
            ["KeyBERT", "DistilBERT", "RAKE", "YAKE"]
        )

        if ModelType == "KeyBERT":
            # kw_model = KeyBERT(model=roberta)

            @st.cache(allow_output_mutation=True)
            def load_model():
                return KeyBERT()

            kw_model = load_model()

        elif ModelType == "RAKE":
            @st.cache(allow_output_mutation=True)
            def load_model():
                return Rake()

            kw_model = load_model()

        elif ModelType == "YAKE":
            @st.cache(allow_output_mutation=True)
            def load_model():
                return yake.KeywordExtractor()

            kw_model = load_model()

        else:
            @st.cache(allow_output_mutation=True)
            def load_model():
                return KeyBERT("distilbert-base-nli-mean-tokens")

            kw_model = load_model()

        top_N = st.slider(
            "# of results",
            min_value=1,
            max_value=30,
            value=10,
            help="You can choose the number of keywords/keyphrases to display. Between 1 and 30, default number is 10.",
        )
        min_Ngrams = st.number_input(
            "Minimum Ngram",
            min_value=1,
            max_value=4,
            help="""The minimum value for the ngram range.

*Keyphrase_ngram_range* sets the length of the resulting keywords/keyphrases.

To extract keyphrases, simply set *keyphrase_ngram_range* to (1, 2) or higher depending on the number of words you would like in the resulting keyphrases.""",
            # help="Minimum value for the keyphrase_ngram_range. keyphrase_ngram_range sets the length of the resulting keywords/keyphrases. To extract keyphrases, simply set keyphrase_ngram_range to (1, # 2) or higher depending on the number of words you would like in the resulting keyphrases.",
        )

        max_Ngrams = st.number_input(
            "Maximum Ngram",
            value=2,
            min_value=1,
            max_value=4,
            help="""The maximum value for the keyphrase_ngram_range.

*Keyphrase_ngram_range* sets the length of the resulting keywords/keyphrases.

To extract keyphrases, simply set *keyphrase_ngram_range* to (1, 2) or higher depending on the number of words you would like in the resulting keyphrases.""",
        )

        StopWordsCheckbox = st.checkbox(
            "Remove stop words",
            help="Tick this box to remove stop words from the document (currently English only)",
        )

        use_MMR = st.checkbox(
            "Use MMR",
            value=True,
            help="You can use Maximal Margin Relevance (MMR) to diversify the results. It creates keywords/keyphrases based on cosine similarity. Try high/low 'Diversity' settings below for interesting variations.",
        )

        Diversity = st.slider(
            "Keyword diversity (MMR only)",
            value=0.5,
            min_value=0.0,
            max_value=1.0,
            step=0.1,
            help="""The higher the setting, the more diverse the keywords.

Note that the *Keyword diversity* slider only works if the *MMR* checkbox is ticked.

""",
        )

    with c2:
        doc = st.text_area(
            "Paste your text below (max 500 words)",
            height=510,
        )

        MAX_WORDS = 500
        import re
        res = len(re.findall(r"\w+", doc))
        if res > MAX_WORDS:
            st.warning(
                "⚠️ Your text contains "
                + str(res)
                + " words."
                + " Only the first 500 words will be reviewed. Stay tuned as increased allowance is coming! 😊"
            )

            doc = doc[:MAX_WORDS]

        submit_button = st.form_submit_button(label="Submit")

    if use_MMR:
        mmr = True
    else:
        mmr = False

    if StopWordsCheckbox:
        StopWords = "english"
    else:
        StopWords = None

if not submit_button:
    st.stop()

if min_Ngrams > max_Ngrams:
    st.warning("min_Ngrams can't be greater than max_Ngrams")
    st.stop()


with st.spinner('Extracting Keywords...'):

    st.header("")
    st.header(f'## 🤖 Model Results: "{ModelType}" ')
    col1, col2, col3 = st.columns(3)

    # Add styling
    cmGreen = sns.light_palette("green", as_cmap=True)
    cmRed = sns.light_palette("red", as_cmap=True)

    if ModelType == "KeyBERT" or ModelType == "DistilBERT":
        keywords = kw_model.extract_keywords(
            doc,
            keyphrase_ngram_range=(min_Ngrams, max_Ngrams),
            use_mmr=mmr,
            stop_words=StopWords,
            top_n=top_N,
            diversity=Diversity,
        )
        with col1:
            th = TextHighlighter(max_ngram_size=max_Ngrams,
                                 highlight_pre="<span style='background-color: #FFFF00; color: black'>", highlight_post="</span>")
            st.header("")
            st.subheader("### 💬 Extracted Keywords: ")
            st.markdown(th.highlight(doc, keywords), unsafe_allow_html=True)

        df = (
            DataFrame(keywords, columns=["Keyword/Keyphrase", "Relevancy"])
            .sort_values(by="Relevancy", ascending=False)
            .reset_index(drop=True)
        )
        df = df.style.background_gradient(
            cmap=cmGreen,
            subset=[
                "Relevancy",
            ],
        )

        with col2:
            st.header("")
            st.subheader("### 🧮 Relevance Scores: ")
            st.table(df)

        with col3:
            st.header("")
            st.subheader("### 📊 Keywords by relevance: ")
            st.bar_chart(df, x="Keyword/Keyphrase", y="Relevancy")

    if ModelType == "RAKE":
        # st.write("RAKE selected")
        kw_model.extract_keywords_from_text(doc)
        keywords = kw_model.get_ranked_phrases_with_scores()

        with col1:
            th = TextHighlighter(max_ngram_size=max_Ngrams,
                                 highlight_pre="<span style='background-color: #FFFF00; color: black'>", highlight_post="</span>")
            st.header("")
            st.subheader("### 💬 Extracted Keywords: ")
            st.markdown(th.highlight(doc, [tuple[1]
                        for tuple in keywords]), unsafe_allow_html=True)
        df = (
            DataFrame(keywords, columns=["Relevancy", "Keyword/Keyphrase"])
            .sort_values(by="Relevancy", ascending=False)
            .reset_index(drop=True)
        )
        # To swap back the position of the columns to ["Keyword/Keyphrase", "Relevancy"].
        df = df[["Keyword/Keyphrase", "Relevancy"]]
        df = df.style.background_gradient(
            cmap=cmGreen,
            subset=[
                "Relevancy",
            ],
        )
        with col2:
            st.header("")
            st.subheader("### 🧮 Relevance Scores: ")
            st.table(df)

        with col3:
            st.header("")
            st.subheader("### 📊 Keywords by relevance: ")
            st.bar_chart(df, x="Keyword/Keyphrase", y="Relevancy")

    if ModelType == "YAKE":
        kw_model = yake.KeywordExtractor(n=max_Ngrams, top=top_N,)
        keywords = kw_model.extract_keywords(doc)

        with col1:
            th = TextHighlighter(max_ngram_size=max_Ngrams,
                                 highlight_pre="<span style='background-color: #FFFF00; color: black'>", highlight_post="</span>")
            st.header("")
            st.markdown("### 💬 Extracted Keywords: ")
            st.markdown(th.highlight(doc, keywords), unsafe_allow_html=True)

        df = (
            DataFrame(keywords, columns=["Keyword/Keyphrase", "Relevancy"])
            .sort_values(by="Relevancy", ascending=True)
            .reset_index(drop=True)
        )
        df = df.style.background_gradient(
            cmap=cmGreen,
            subset=[
                "Relevancy",
            ],
        )
        with col2:
            st.header("")
            st.markdown("### 🧮 Relevance Scores: ")
            st.info(
                'Lower Score indicates better relevance in the "YAKE" model.', icon="ℹ️")
            st.table(df)

        with col3:
            st.header("")
            st.subheader("### 📊 Keywords by relevance: ")
            st.bar_chart(df, x="Keyword/Keyphrase", y="Relevancy")

st.header("")
