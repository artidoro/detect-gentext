import os
import json

import numpy as np
from sklearn.utils import shuffle
import pandas as pd

def _load_split(data_dir, source, split, n=np.inf, start_at=0):
    path = os.path.join(data_dir, f'{source}.{split}.jsonl')
    if not os.path.exists(path):
        path = os.path.join(data_dir, f'{source}.jsonl')
        assert os.path.exists(path), f'source {source} not found.'
    texts = []
    for i, line in enumerate(open(path)):
        if i < start_at:
            continue
        if i >= start_at + n:
            break
        d = json.loads(line)
        if 'text' in d:
            texts.append(d['text'])
        elif 'article' in d:
            texts.append(d['article'])
    return texts

def _load_split_grover(data_dir, source, split, n=np.inf):
    path = os.path.join(data_dir, f'{source}.{split}.jsonl')
    if not os.path.exists(path):
        path = os.path.join(data_dir, f'{source}.jsonl')
        assert os.path.exists(path), f'source {path} not found.'
    texts = []
    labels = []
    for i, line in enumerate(open(path)):
        if i >= n:
            break
        elt = json.loads(line)
        if 'text' in elt:
            if len(elt['text'].split()) > 0:
                texts.append(elt['text'])
            else:
                continue
        elif 'article' in elt:
            if len(elt['article'].split()) > 0:
                texts.append(elt['article'])
            else:
                continue
        labels.append(elt['label'] == 'machine' or elt['label'] == 'generated')
    return texts, labels

def load_split(data_dir, sources, split, n=np.inf):
    texts = []
    labels = []
    webtext_read = 0
    webtext_store = []
    for source in sources.split(';'):
        if len(source) == 0:
            continue
        # openai outputs (combinations of webtext and generated from two different files)
        if not ('generator' in source or 'gpt' in source or 'covid' in source): #or 'gpt2' in source or 'gptneo' in source or 'gpt-j' in source or 'gpt-neo' in source 
            webtext = _load_split(data_dir, 'webtext', split, n=n//2, start_at=webtext_read)
            gen = _load_split(data_dir, source, split, n=n//2)
            t = webtext+gen
            l = [0]*len(webtext)+[1]*len(gen)
            webtext_read += len(webtext)
            webtext_store += webtext
        # everything else (grover, covid generated text etc. single file with both generated and pristine)
        else:
            t, l = _load_split_grover(data_dir, source, split, n)
        print(f'Loaded {len(t)} datapoints from {source}.')
        texts += t
        labels += l

    data = {
        'text': texts,
        'labels': labels
    }
    df = pd.DataFrame(data=data)
    if split == 'train':
        df = shuffle(df)

    return df