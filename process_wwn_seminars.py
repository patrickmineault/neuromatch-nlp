import collections
import itertools
import json
import pandas as pd
from sentence_transformers import SentenceTransformer, util

def clean_abstract(abstract):
    if abstract.startswith('Host:') or abstract.startswith("// Flashtalk") or abstract.startswith('no'):
        return ""
    return abstract

def clean_title(title):
    if title.startswith('NMC4'):
        return title.split(':')[-1]
    elif title.startswith('TBD'):
        return ""
    return title

def main():
    with open('wwn-seminars.json') as f:
        data = json.load(f)

    # Start by cleaning up the datasets and removing duplicates.
    cleaned_data = []
    for row in data.values():
        cleaned_data.append(
            {'tags': row['topic_tags'],
             'author': row['seminar_speaker'],
             'title': clean_title(row['seminar_title']),
             'abstract': clean_abstract(row['seminar_abstract']),
             'id': row['partition_key'],
             }
        )

    df = pd.DataFrame(cleaned_data)
    df['text'] = df.title + " [SEP] " + df.abstract
    df['duplicate'] = False

    # Remove close to exact duplicates
    model = SentenceTransformer('average_word_embeddings_glove.6B.300d')
    encodings = model.encode(df.text, show_progress_bar=False)

    for i in range(len(encodings)):
        search_hits = util.semantic_search(encodings[i], encodings[i+1:])

        # Verify if there are duplicates
        for j in range(len(search_hits[0])):
            if search_hits[0][j]['score'] > .995:
                # Exact duplicate
                # Mark as such
                bad_location = search_hits[0][j]['corpus_id'] + i + 1
                df.loc[bad_location, 'duplicate'] = True
            else:
                break

    # Create a tag whitelist
    #df = df.loc[df.tags.map(lambda x: len(x) > 0)]

    all_tags = itertools.chain(*df.tags.values)
    all_tags = collections.Counter(all_tags)

    # Single tags can't help us here.
    all_top = [k for k, v in all_tags.items() if v > 1]

    # Also tag bad entries.
    df["bad_entry"] = ((df.title == "") & (df.abstract == "")) | (df.tags.map(lambda x: len([a for a in x if a in all_top]))==0)
    print(f"{len(encodings)} documents processed, {df.duplicate.sum()} duplicates identified, {df.bad_entry.sum()} bad entries")
    df[(~df.bad_entry) & (~df.duplicate)].to_json("wwn_seminars_cleaned.json")

if __name__ == '__main__':
    main()