import neuromatch
import numpy as np
import pandas as pd
import retrieve_documents

def main():
    df = pd.read_csv("data/transformed/sfn_2015.csv")

    # Tune for only 100 abstracts.
    targets = np.argsort(np.random.rand(df.shape[0]))[:100]
    df = df.iloc[targets]

    tops = []
    for vectorizer in ['entropy', 'bm25', 'count', 'tfidf']:
        for n_components in [5, 7, 9, 11, 13, 15, 18, 30, 50, 85, 99]:
            paper_texts = df.text.values
            # We use the TF-IDF encoder that was pioneered by Neuromatch
            encodings = neuromatch.compute_embeddings(
                paper_texts, n_components=n_components, min_df=3, max_df=0.85, weighting=vectorizer, projection='pca'
            )

            top = retrieve_documents.get_top(encodings, df)
            for t in top:
                t['model_name'] = 'neuromatch'
                t['n_components'] = n_components
                t['vectorizer'] = vectorizer

            print(n_components, vectorizer)
            print(np.mean([x['score_1'] for x in top]))
            print(np.mean([x['score_5'] for x in top]))
            print(np.mean([x['score_10'] for x in top]))

            tops.append(top)
        
    pd.DataFrame(tops).to_csv(f'small_neuromatch_tuning.csv')

if __name__ == "__main__":
    main()