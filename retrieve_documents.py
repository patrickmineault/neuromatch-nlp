import click
import neuromatch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

def calculate_overlap(t1, t2):
    parts1 = t1.split('.')[:3]
    parts2 = t2.split('.')[:3]
    overlap = 0
    for p0, p1 in zip(parts1, parts2):
        if p0 == p1:
            overlap += 1
        else:
            break

    return overlap

def get_top(encodings, df):
    tops = []

    use_tags = False
    if 'tags' in df.columns:
        use_tags = True

    for search_id in range(df.shape[0]):
        doc = df.iloc[search_id]
        search_hits = util.semantic_search(encodings[search_id], encodings, top_k=11)[0][1:]
        # Take the top 5 hits

        # Calculate tag overlap
        retrieval = {'probe_id': doc.id, 'probe_title': doc.title, 'probe_abstract': doc.abstract}
        
        if use_tags:
            tags = set(doc.tags)
            f1 = 0
            for i, t in enumerate(search_hits):
                doc2 = df.iloc[t['corpus_id']]
                tags2 = set(doc2['tags'])
                f1 += len(tags.intersection(tags2)) / (1/2 * (.01 + len(tags) + len(tags2)))
                retrieval[f'id_{i}'] = doc2.id
                retrieval[f'title_{i}'] = doc2.title
                retrieval[f'abstract_{i}'] = doc2.abstract

                if i in (0, 4, 9):
                    retrieval[f'score_{i+1}'] = f1 / (i + 1)
        else:
            # Use topic overlap
            topic = doc.topic
            overlap = 0
            for i, t in enumerate(search_hits):
                doc2 = df.iloc[t['corpus_id']]
                topic2 = doc2['topic']
                overlap += calculate_overlap(topic, topic2)
                retrieval[f'id_{i}'] = doc2.id
                retrieval[f'title_{i}'] = doc2.title
                retrieval[f'abstract_{i}'] = doc2.abstract

                if i in (0, 4, 9):
                    retrieval[f'score_{i+1}'] = overlap / (i + 1)

        tops.append(retrieval)

    return tops

@click.command()
@click.option('--dataset')
def main(dataset):
    if dataset == 'wwn':
        df = pd.read_json("wwn_seminars_cleaned.json")
    elif dataset == 'sfn_2015_sample':
        df = pd.read_csv("data/transformed/sfn_2015_subsample.csv")
    elif dataset == 'sfn_2015':
        df = pd.read_csv("data/transformed/sfn_2015.csv")
    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented")

    model_names = ['neuromatch', 'average_word_embeddings_glove.6B.300d', 'all-mpnet-base-v2', 'allenai-specter']

    tops = []
    for model_name in model_names:
        # Based on AllenAI specter
        # https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/semantic-search/semantic_search_publications.py
        paper_texts = df.text.values
        if model_name == 'neuromatch':
            # We use the TF-IDF encoder that was pioneered by Neuromatch
            encodings = neuromatch.compute_embeddings(
                paper_texts, n_components=250, min_df=3, max_df=0.85, weighting='tfidf', projection='pca'
            )
        else:
            model = SentenceTransformer(model_name)
            encodings = model.encode(paper_texts, show_progress_bar=True)

        top = get_top(encodings, df)
        for t in top:
            t['model_name'] = model_name

        tops += top
        print(model_name)
        print(np.mean([x['score_1'] for x in top]))
        print(np.mean([x['score_5'] for x in top]))
        print(np.mean([x['score_10'] for x in top]))
        
    pd.DataFrame(tops).to_csv(f'{dataset}_rankings.csv')


if __name__ == '__main__':
    main()