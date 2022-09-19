from ast import keyword
import streamlit as st
import numpy as np
from pandas import DataFrame
from keybert import KeyBERT
# For Flair (Keybert)
from flair.embeddings import TransformerDocumentEmbeddings
import seaborn as sns
# For RAKE Algo
from rake_nltk import Rake

import nltk
nltk.download('stopwords')
nltk.download('punkt')

import os
import json


st.set_page_config(
    page_title="DAEN 690 Team Concorde: NLP Algorithms",
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


_max_width_()

c30, c31, c32 = st.columns([2.5, 1, 3])

with c30:
    # st.image("logo.png", width=400)
    st.title("DAEN 690 Team Concorde: NLP Algorithms")
    st.header("")


st.markdown("")
st.markdown("## 📌 Paste text")
with st.form(key="my_form"):

    ce, c1, ce, c2, c3 = st.columns([0.07, 1, 0.07, 5, 0.07])
    with c1:
        ModelType = st.radio(
            "Choose your model",
            ["KeyBERT (Default)", "DistilBERT", "RAKE"],
            help="At present, you can choose between 2 models (Flair or DistilBERT) to embed your text. More to come!",
        )

        if ModelType == "KeyBERT (Default)":
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

if ModelType == "KeyBERT (Default)" or ModelType == "DistilBERT":
    keywords = kw_model.extract_keywords(
        doc,
        keyphrase_ngram_range=(min_Ngrams, max_Ngrams),
        use_mmr=mmr,
        stop_words=StopWords,
        top_n=top_N,
        diversity=Diversity,
    )
    st.write("Keywords: " + str(keywords))
    df = (
        DataFrame(keywords, columns=["Keyword/Keyphrase", "Relevancy"])
        .sort_values(by="Relevancy", ascending=False)
        .reset_index(drop=True)
    )

if ModelType == "RAKE":
    st.write("RAKE selected")
    kw_model.extract_keywords_from_text(doc)
    # st.write("RESULT" + result)
    keywords = kw_model.get_ranked_phrases_with_scores()
    st.write("Keywords: " + str(keywords))
    df = (
        DataFrame(keywords, columns=["Relevancy", "Keyword/Keyphrase"])
        .sort_values(by="Relevancy", ascending=False)
        .reset_index(drop=True)
    )
    # To swap back the position of the columns to ["Keyword/Keyphrase", "Relevancy"].
    df = df[["Keyword/Keyphrase", "Relevancy"]]

st.markdown("## Results")

st.header("")


print(df)
df.index += 1

# Add styling
cmGreen = sns.light_palette("green", as_cmap=True)
cmRed = sns.light_palette("red", as_cmap=True)
df = df.style.background_gradient(
    cmap=cmGreen,
    subset=[
        "Relevancy",
    ],
)

c1, c2, c3 = st.columns([1, 3, 1])

# format_dictionary = {
#     "Relevancy": "{:.1%}",
# }

# df = df.format(format_dictionary)

with c2:
    st.write(df)
# st.table(df)
