import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(layout="wide")

from urllib.error import URLError

@st.cache
def get_ranking_data(dataset):
    df = pd.read_csv(f"data/outputs/{dataset}_sample_rankings.csv")
    return df

try:
    dataset_id = 'sfn_2015'
    df = get_ranking_data(dataset_id)
    print("done")
    #model_names = df.model_name.unique()
    model_names = ['neuromatch', 'all-mpnet-base-v2', 'allenai-specter', 't4yt_trained', 'aggregate']

    probes = df.probe_id.unique()

    document_id = st.selectbox(
        "Choose document", df['probe_title'].unique(), 78
    )

    if not document_id:
        st.error("Please select a document")
    else:
        document_id = df.loc[df.probe_title == document_id].iloc[0].probe_id
        data = df.loc[df.probe_id == document_id]
        st.write("#### " + data.iloc[0].probe_title)
        st.write(data.iloc[0].probe_abstract)

        st.write("##### Most similar documents")

        cols = st.columns(len(model_names))
        for i, col in enumerate(cols):
            col.write("###### " + model_names[i][:23])
            row = data.loc[data.model_name == model_names[i]].iloc[0]
            col.write(f"Scores (1/5/10): {row['score_1']:.1f}, {row['score_5']:.1f}, {row['score_10']:.1f}")
            for j in range(10):
                title = row[f"title_{j}"]
                if title is None:
                    title = "NA"
                if isinstance(title, float) and np.isnan(title):
                    title = "NA"
                expander = col.expander(title)
                expander.write(row[f"abstract_{j}"])

except URLError as e:
    st.error(
        """
        **This demo requires internet access.**

        Connection error: %s
    """
        % e.reason
    )
