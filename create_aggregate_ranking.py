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

def get_top(df_rankings, df, models):
    df_rankings = df_rankings.loc[df_rankings.model_name.isin(models)]
    tops = []

    use_tags = False
    if 'tags' in df.columns:
        use_tags = True

    for search_id in range(df.shape[0]):
        # Use reranking on the target models
        doc = df.iloc[search_id]
        probe_id = doc.id
        rankings = df_rankings.query(f'probe_id == {probe_id}').filter(regex='id_[0-9]+', axis=1).values.ravel()
        len_rankings = rankings.size // len(models)
        base_scores = np.arange(len_rankings, 0, -1)
        base_scores = np.concatenate([base_scores] * len(models))
        scores = [rankings, base_scores]
        df_reranked = pd.DataFrame(scores).T
        df_reranked.columns = ['corpus_id', 'score']
        df_reranked = df_reranked.groupby('corpus_id').sum().sort_values('score', ascending=False)
        search_hits = df_reranked.reset_index()
        search_hits.corpus_id = search_hits.corpus_id.map(lambda x: str(x))
        search_hits = search_hits.head(10)
        
        if search_hits.shape[0] != 10:
            print(df.id.max())
            print(df.id.min())
            print(df.shape)
            print(probe_id)
            print(search_hits)
            print(df_rankings.probe_id.min())
            print(df_rankings.probe_id.max())
            assert False

        # Take the top 5 hits

        # Calculate tag overlap
        retrieval = {'probe_id': doc.id, 'probe_title': doc.title, 'probe_abstract': doc.abstract}
        
        if use_tags:
            tags = set(doc.tags)
            f1 = 0
            for i, t in search_hits.iterrows():
                print(t)
                print(t['corpus_id'])
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
            
            for i, (_, t) in enumerate(search_hits.iterrows()):
                doc2 = df.loc[df.id == int(float(t['corpus_id']))]
                assert doc2.shape[0]
                doc2 = doc2.iloc[0]
                topic2 = doc2['topic']
                overlap += calculate_overlap(topic, topic2)
                retrieval[f'id_{i}'] = doc2.id
                retrieval[f'title_{i}'] = doc2.title
                retrieval[f'abstract_{i}'] = doc2.abstract

                if i in (0, 4, 9):
                    retrieval[f'score_{i+1}'] = overlap / (i + 1)

        tops.append(retrieval)

    return tops

def main():
    dataset = 'sfn_2015'
    df = pd.read_csv("data/transformed/sfn_2015.csv")
    df_rankings = pd.read_csv('data/outputs/sfn_2015_all_rankings.csv')
    models = ['neuromatch', 'all-mpnet-base-v2', 't4yt_trained', 'allenai-specter']

    top = get_top(df_rankings, df, models)
    print(np.mean([x['score_1'] for x in top]))
    print(np.mean([x['score_5'] for x in top]))
    print(np.mean([x['score_10'] for x in top]))
    pd.DataFrame(top).to_csv(f'{dataset}_rerankings.csv')


if __name__ == '__main__':
    main()