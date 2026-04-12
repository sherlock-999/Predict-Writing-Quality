"""
Text Statistics
===============
Compute word, sentence, and paragraph features from the reconstructed essay text.

All three functions receive a plain string (the reconstructed essay) and return
a flat dict of feature_name → value ready to merge into the per-essay row.

Processing order inside compute_features():
    Step 4a : word_features(text)
    Step 4b : sentence_features(text)
    Step 4c : paragraph_features(text)
"""

import re
import numpy as np


# ── Private helper ────────────────────────────────────────────────────────────

def _agg(values: list, prefix: str) -> dict:
    """
    Compute descriptive stats for a list of numeric values.
    Returns a dict with keys: {prefix}_count/mean/min/max/first/last/q1/median/q3/sum.
    All values default to 0.0 when the list is empty.
    """
    if not values:
        names = ['count','mean','min','max','first','last','q1','median','q3','sum']
        return {f'{prefix}_{n}': 0.0 for n in names}
    arr = np.array(values, dtype=float)
    return {
        f'{prefix}_count':  len(arr),
        f'{prefix}_mean':   arr.mean(),
        f'{prefix}_min':    arr.min(),
        f'{prefix}_max':    arr.max(),
        f'{prefix}_first':  arr[0],
        f'{prefix}_last':   arr[-1],
        f'{prefix}_q1':     np.percentile(arr, 25),
        f'{prefix}_median': np.median(arr),
        f'{prefix}_q3':     np.percentile(arr, 75),
        f'{prefix}_sum':    arr.sum(),
    }


# ── Public feature functions ──────────────────────────────────────────────────

def word_features(text: str) -> dict:
    """
    Word-level features from the reconstructed essay.

    Words are split on spaces, newlines, and sentence-ending punctuation.
    Each word's character length is measured (the 'q' placeholder counts as
    one character per original letter, so lengths are representative).

    Features
    --------
    word_len_{count,mean,min,max,first,last,q1,median,q3,sum}
    """
    raw_words = [w for w in re.split(r'[ \n\.\?\!]', text) if w]
    lengths   = [len(w) for w in raw_words]
    feats     = _agg(lengths, 'word_len')
    # remove word_len_min — always 1 or 0, no signal
    feats.pop('word_len_min', None)
    return feats


def sentence_features(text: str) -> dict:
    """
    Sentence-level features from the reconstructed essay.

    Sentences are split on [.?!]. Newlines within a sentence are stripped
    before measuring length and word count.

    Features
    --------
    sent_count
    sent_len_{mean,min,max,first,last,q1,median,q3,sum}
    sent_word_count_{mean,max,first,last,q1,median,q3}
    """
    raw_sents = [s.replace('\n', '').strip()
                 for s in re.split(r'[\.\?\!]', text)
                 if s.strip()]

    sent_lens = [len(s)       for s in raw_sents]
    sent_wc   = [len(s.split()) for s in raw_sents]

    feats = {'sent_count': len(raw_sents)}

    sent_len_feats = _agg(sent_lens, 'sent_len')
    sent_len_feats.pop('sent_len_count', None)   # already stored as sent_count
    feats.update(sent_len_feats)

    sent_wc_feats = _agg(sent_wc, 'sent_word_count')
    sent_wc_feats.pop('sent_word_count_count', None)
    # drop sent_word_count_min/sum — gain < 15
    sent_wc_feats.pop('sent_word_count_min', None)
    sent_wc_feats.pop('sent_word_count_sum', None)
    feats.update(sent_wc_feats)

    return feats


def paragraph_features(text: str) -> dict:
    """
    Paragraph-level features from the reconstructed essay.

    Paragraphs are split on newline characters ('\\n'). Empty lines are ignored.

    Features
    --------
    paragraph_count
    paragraph_len_{mean,min,max,first,last,q1,median,q3,sum}
    paragraph_word_count_{mean,min,max,first,last,q1,median,q3,sum}
    """
    raw_paras = [p.strip() for p in text.split('\n') if p.strip()]

    para_lens = [len(p)       for p in raw_paras]
    para_wc   = [len(p.split()) for p in raw_paras]

    feats = {'paragraph_count': len(raw_paras)}

    para_len_feats = _agg(para_lens, 'paragraph_len')
    para_len_feats.pop('paragraph_len_count', None)
    feats.update(para_len_feats)

    para_wc_feats = _agg(para_wc, 'paragraph_word_count')
    para_wc_feats.pop('paragraph_word_count_count', None)
    feats.update(para_wc_feats)

    return feats
