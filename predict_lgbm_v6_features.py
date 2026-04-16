"""
Inference: Predict Essay Scores — Kaggle Submission (v6 features)
==================================================================
Self-contained script: all feature engineering logic is inlined.
No local module imports required.

Feature set: v6_features (226 features = v5 206 + 20 new)
  v5 base (206):
    - Extended activity counts:    Replace, Paste
    - Extended text_change counts: doublequote, semicolon, equals, slash, backslash, colon
    - Extended down_event counts:  ArrowDown, ArrowUp, singlequote, Delete, Unidentified
    - Full up_event counts (16 events)
    - n_unique_activity, n_unique_up_event
    - input_word_length_median
    - action_time_min, up_time_{mean,std,median,q1,q3}
    - cursor_position_min, word_count_{min,mean,max,sum}
    - Per-word normalisations for all new counts
  v6 new (20):
    - Event TF-IDF SVD: word 1–5-gram TF-IDF on down_event sequence → TruncatedSVD(20)

Two TF-IDF SVD pipelines must be loaded at inference time:
  TFIDF_PATH       — char 2–4-gram on reconstructed essay text (same as v4/v5)
  EVENT_TFIDF_PATH — word 1–5-gram on down_event sequence (v6 new)

Both are saved during precompute_features_v6.py and uploaded to Kaggle as a
single dataset: writing-quality-tfidf-v6

Paths (Kaggle notebook environment):
    DATA_DIR         = /kaggle/input/competitions/linking-writing-processes-to-writing-quality
    MODELS_DIR       = /kaggle/input/datasets/sherlocked999/writing-quality-lightgbm-v6
    TFIDF_PATH       = /kaggle/input/datasets/sherlocked999/writing-quality-tfidf-v6/tfidf_svd.pkl
    EVENT_TFIDF_PATH = /kaggle/input/datasets/sherlocked999/writing-quality-tfidf-v6/event_tfidf_svd.pkl
    OUTPUT_PATH      = submission.csv
"""

import re
import glob
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import skew as _skew

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR         = '/kaggle/input/competitions/linking-writing-processes-to-writing-quality'
MODELS_DIR       = '/kaggle/input/datasets/sherlocked999/writing-quality-lightgbm-v6'
TFIDF_PATH       = '/kaggle/input/datasets/sherlocked999/writing-quality-tfidf-v6/tfidf_svd.pkl'
EVENT_TFIDF_PATH = '/kaggle/input/datasets/sherlocked999/writing-quality-tfidf-v6/event_tfidf_svd.pkl'
OUTPUT_PATH      = 'submission.csv'

# ── TF-IDF config (must match training) ──────────────────────────────────────
TFIDF_N_COMPONENTS       = 20
TFIDF_SVD_COLS           = [f'tfidf_svd_{i}'       for i in range(TFIDF_N_COMPONENTS)]
EVENT_TFIDF_N_COMPONENTS = 20
EVENT_TFIDF_SVD_COLS     = [f'event_tfidf_svd_{i}' for i in range(EVENT_TFIDF_N_COMPONENTS)]


# =============================================================================
# FEATURE COLUMNS  (v6 — 226 features)
# =============================================================================

FEATURE_COLS = [

    # ── 1. WORD CHARACTERISTICS ───────────────────────────────────────────────
    'word_len_sum',
    'word_len_mean',
    'word_len_q3',
    'word_len_max',
    'word_len_first',
    'word_len_last',
    'word_len_count',
    'input_word_count',
    'input_word_length_mean',
    'input_word_length_max',
    'input_word_length_std',
    'input_word_length_median',
    'input_word_length_skew',
    'word_count_min',
    'word_count_mean',
    'word_count_max',
    'word_count_sum',
    'word_count_std',
    'word_count_median',
    'word_count_q1',
    'word_count_q3',

    # ── 2. SENTENCE CHARACTERISTICS ───────────────────────────────────────────
    'sent_len_mean',
    'sent_len_median',
    'sent_len_min',
    'sent_len_max',
    'sent_len_first',
    'sent_len_last',
    'sent_len_q1',
    'sent_len_q3',
    'sent_per_word',
    'sent_word_count_mean',
    'sent_word_count_max',
    'sent_word_count_first',
    'sent_word_count_last',
    'sent_word_count_q1',
    'sent_word_count_median',
    'sent_word_count_q3',

    # ── 3. PARAGRAPH CHARACTERISTICS ──────────────────────────────────────────
    'paragraph_len_mean',
    'paragraph_len_median',
    'paragraph_len_min',
    'paragraph_len_max',
    'paragraph_len_first',
    'paragraph_len_last',
    'paragraph_len_q1',
    'paragraph_len_q3',
    'paragraph_count',
    'paragraph_word_count_mean',
    'paragraph_word_count_min',
    'paragraph_word_count_max',
    'paragraph_word_count_first',
    'paragraph_word_count_last',
    'paragraph_word_count_q1',
    'paragraph_word_count_median',
    'paragraph_word_count_q3',

    # ── 4. PUNCTUATION & FORMATTING ───────────────────────────────────────────
    'comma_per_word',
    'comma_text_per_word',
    'period_per_word',
    'period_text_per_word',
    'enter_per_word',
    'newline_per_word',
    'shift_per_word',
    'capslock_per_word',
    'dash_per_word',
    'question_per_word',
    'quote_per_word',
    'doublequote_text_per_word',
    'semicolon_text_per_word',
    'equals_text_per_word',
    'slash_text_per_word',
    'backslash_text_per_word',
    'colon_text_per_word',
    'singlequote_per_word',

    # ── 5. TYPING SPEED & FLUENCY ─────────────────────────────────────────────
    'keys_per_second',
    'action_time_min',
    'action_time_sum',
    'action_time_mean',
    'action_time_std',
    'action_time_median',
    'action_time_max',
    'action_time_q1',
    'action_time_q3',
    'q_keys_per_word',
    'q_text_per_word',
    'space_per_word',
    'space_text_per_word',
    'n_unique_activity',
    'n_unique_text_change',
    'n_unique_down_event',
    'n_unique_up_event',

    # ── 6. PAUSING BEHAVIOR ───────────────────────────────────────────────────
    'idle_largest_latency',
    'idle_median_latency',
    'idle_mean',
    'idle_std',
    'idle_total',
    'pauses_half_per_word',
    'pauses_1_per_word',
    'pauses_1half_per_word',
    'pauses_2_per_word',
    'pauses_3_per_word',

    # ── 7. REVISION BEHAVIOR ──────────────────────────────────────────────────
    'removecut_per_word',
    'backspace_per_word',
    'delete_per_word',
    'r_burst_mean',
    'r_burst_std',
    'r_burst_median',
    'r_burst_max',
    'r_burst_first',

    # ── 8. PRODUCTION FLOW & NAVIGATION ──────────────────────────────────────
    'input_per_word',
    'replace_per_word',
    'paste_per_word',
    'nonproduction_per_word',
    'p_burst_per_word',
    'p_burst_mean',
    'p_burst_std',
    'p_burst_median',
    'p_burst_max',
    'p_burst_first',
    'p_burst_last',
    'product_to_keys',
    'session_duration_sec',
    'cursor_position_min',
    'cursor_position_mean',
    'cursor_position_std',
    'cursor_position_median',
    'cursor_position_q1',
    'cursor_position_q3',
    'cursor_position_max',
    'leftclick_per_word',
    'arrowleft_per_word',
    'arrowright_per_word',
    'arrowdown_per_word',
    'arrowup_per_word',
    'unidentified_per_word',
    'down_time_min',
    'down_time_mean',
    'down_time_std',
    'down_time_median',
    'down_time_q1',
    'down_time_q3',
    'down_time_max',
    'up_time_min',
    'up_time_mean',
    'up_time_std',
    'up_time_median',
    'up_time_q1',
    'up_time_q3',
    'up_time_max',

    # ── 9. UP EVENT COUNTS ────────────────────────────────────────────────────
    'up_event_q_cnt',
    'up_event_Space_cnt',
    'up_event_Backspace_cnt',
    'up_event_Shift_cnt',
    'up_event_ArrowRight_cnt',
    'up_event_Leftclick_cnt',
    'up_event_ArrowLeft_cnt',
    'up_event_period_cnt',
    'up_event_comma_cnt',
    'up_event_ArrowDown_cnt',
    'up_event_ArrowUp_cnt',
    'up_event_Enter_cnt',
    'up_event_CapsLock_cnt',
    'up_event_singlequote_cnt',
    'up_event_Delete_cnt',
    'up_event_Unidentified_cnt',

    # ── 10. RAW ACTIVITY / EVENT COUNTS ──────────────────────────────────────
    'activity_Replace_cnt',
    'activity_Paste_cnt',
    'text_change_doublequote_cnt',
    'text_change_semicolon_cnt',
    'text_change_equals_cnt',
    'text_change_slash_cnt',
    'text_change_backslash_cnt',
    'text_change_colon_cnt',
    'down_event_ArrowDown_cnt',
    'down_event_ArrowUp_cnt',
    'down_event_singlequote_cnt',
    'down_event_Delete_cnt',
    'down_event_Unidentified_cnt',

    # ── 11. READABILITY & STRUCTURAL CONSISTENCY ─────────────────────────────
    'ari_score',
    'coleman_liau_score',
    'sent_len_std',
    'sent_len_cv',
    'para_len_std',
    'para_len_cv',
    'para_balance',
    'words_per_minute',

    # ── 12. REVISION TIMING ───────────────────────────────────────────────────
    'revision_ratio_early',
    'revision_ratio_mid',
    'revision_ratio_late',
    'revision_timing_mean',
    'revision_timing_std',

    # ── 13. TEXT TF-IDF SVD  (char 2–4-gram on reconstructed essay) ──────────
    *TFIDF_SVD_COLS,         # tfidf_svd_0 … tfidf_svd_19

    # ── 14. EVENT TF-IDF SVD  (word 1–5-gram on down_event sequence) ─────────
    *EVENT_TFIDF_SVD_COLS,   # event_tfidf_svd_0 … event_tfidf_svd_19
]


# =============================================================================
# FEATURE ENGINEERING  (all logic inlined — no local imports)
# =============================================================================

# ── Step 1: Count features ────────────────────────────────────────────────────

def _count_features(essay: pd.DataFrame) -> dict:
    act = essay['activity']
    tc  = essay['text_change']
    de  = essay['down_event']
    ue  = essay['up_event']

    feats = {}

    for value, key in [
        ('Input',         'activity_Input_cnt'),
        ('Remove/Cut',    'activity_RemoveCut_cnt'),
        ('Nonproduction', 'activity_Nonproduction_cnt'),
        ('Replace',       'activity_Replace_cnt'),
        ('Paste',         'activity_Paste_cnt'),
    ]:
        feats[key] = int((act == value).sum())

    for value, key in [
        ('q',   'text_change_q_cnt'),
        (' ',   'text_change_space_cnt'),
        ('.',   'text_change_period_cnt'),
        (',',   'text_change_comma_cnt'),
        ('\n',  'text_change_newline_cnt'),
        ("'",   'text_change_quote_cnt'),
        ('-',   'text_change_dash_cnt'),
        ('?',   'text_change_question_cnt'),
        ('"',   'text_change_doublequote_cnt'),
        (';',   'text_change_semicolon_cnt'),
        ('=',   'text_change_equals_cnt'),
        ('/',   'text_change_slash_cnt'),
        ('\\',  'text_change_backslash_cnt'),
        (':',   'text_change_colon_cnt'),
    ]:
        feats[key] = int((tc == value).sum())

    for value, key in [
        ('q',            'down_event_q_cnt'),
        ('Space',        'down_event_Space_cnt'),
        ('Backspace',    'down_event_Backspace_cnt'),
        ('Shift',        'down_event_Shift_cnt'),
        ('ArrowRight',   'down_event_ArrowRight_cnt'),
        ('Leftclick',    'down_event_Leftclick_cnt'),
        ('ArrowLeft',    'down_event_ArrowLeft_cnt'),
        ('.',            'down_event_period_cnt'),
        (',',            'down_event_comma_cnt'),
        ('Enter',        'down_event_Enter_cnt'),
        ('CapsLock',     'down_event_CapsLock_cnt'),
        ('ArrowDown',    'down_event_ArrowDown_cnt'),
        ('ArrowUp',      'down_event_ArrowUp_cnt'),
        ("'",            'down_event_singlequote_cnt'),
        ('Delete',       'down_event_Delete_cnt'),
        ('Unidentified', 'down_event_Unidentified_cnt'),
    ]:
        feats[key] = int((de == value).sum())

    for value, key in [
        ('q',            'up_event_q_cnt'),
        ('Space',        'up_event_Space_cnt'),
        ('Backspace',    'up_event_Backspace_cnt'),
        ('Shift',        'up_event_Shift_cnt'),
        ('ArrowRight',   'up_event_ArrowRight_cnt'),
        ('Leftclick',    'up_event_Leftclick_cnt'),
        ('ArrowLeft',    'up_event_ArrowLeft_cnt'),
        ('.',            'up_event_period_cnt'),
        (',',            'up_event_comma_cnt'),
        ('ArrowDown',    'up_event_ArrowDown_cnt'),
        ('ArrowUp',      'up_event_ArrowUp_cnt'),
        ('Enter',        'up_event_Enter_cnt'),
        ('CapsLock',     'up_event_CapsLock_cnt'),
        ("'",            'up_event_singlequote_cnt'),
        ('Delete',       'up_event_Delete_cnt'),
        ('Unidentified', 'up_event_Unidentified_cnt'),
    ]:
        feats[key] = int((ue == value).sum())

    feats['n_unique_activity']    = int(act.nunique())
    feats['n_unique_text_change'] = int(tc.nunique())
    feats['n_unique_down_event']  = int(de.nunique())
    feats['n_unique_up_event']    = int(ue.nunique())

    return feats


# ── Step 2: Input word features ───────────────────────────────────────────────

def _input_word_features(essay: pd.DataFrame) -> dict:
    plain_input = essay[
        ~essay['text_change'].str.contains('=>', na=False) &
        (essay['text_change'] != 'NoChange')
    ]['text_change'].str.cat(sep='')

    word_seqs = re.findall(r'q+', plain_input)
    lengths   = [len(w) for w in word_seqs]

    return {
        'input_word_count':         len(lengths),
        'input_word_length_mean':   np.mean(lengths)      if lengths else 0.0,
        'input_word_length_max':    np.max(lengths)       if lengths else 0.0,
        'input_word_length_std':    np.std(lengths)       if lengths else 0.0,
        'input_word_length_median': np.median(lengths)    if lengths else 0.0,
        'input_word_length_skew':   float(_skew(lengths)) if lengths else 0.0,
    }


# ── Step 3: Timing features ───────────────────────────────────────────────────

def _timing_features(essay: pd.DataFrame) -> dict:
    feats = {}

    at = essay['action_time']
    feats['action_time_min']    = float(at.min())
    feats['action_time_sum']    = float(at.sum())
    feats['action_time_mean']   = float(at.mean())
    feats['action_time_std']    = float(at.std())
    feats['action_time_median'] = float(at.median())
    feats['action_time_max']    = float(at.max())
    feats['action_time_q1']     = float(at.quantile(0.25))
    feats['action_time_q3']     = float(at.quantile(0.75))

    dt = essay['down_time']
    feats['down_time_min']    = float(dt.min())
    feats['down_time_mean']   = float(dt.mean())
    feats['down_time_std']    = float(dt.std())
    feats['down_time_median'] = float(dt.median())
    feats['down_time_q1']     = float(dt.quantile(0.25))
    feats['down_time_q3']     = float(dt.quantile(0.75))
    feats['down_time_max']    = float(dt.max())

    ut = essay['up_time']
    feats['up_time_min']    = float(ut.min())
    feats['up_time_mean']   = float(ut.mean())
    feats['up_time_std']    = float(ut.std())
    feats['up_time_median'] = float(ut.median())
    feats['up_time_q1']     = float(ut.quantile(0.25))
    feats['up_time_q3']     = float(ut.quantile(0.75))
    feats['up_time_max']    = float(ut.max())

    cp = essay['cursor_position']
    feats['cursor_position_min']    = float(cp.min())
    feats['cursor_position_mean']   = float(cp.mean())
    feats['cursor_position_std']    = float(cp.std())
    feats['cursor_position_median'] = float(cp.median())
    feats['cursor_position_q1']     = float(cp.quantile(0.25))
    feats['cursor_position_q3']     = float(cp.quantile(0.75))
    feats['cursor_position_max']    = float(cp.max())

    wc = essay['word_count']
    feats['word_count_min']    = float(wc.min())
    feats['word_count_mean']   = float(wc.mean())
    feats['word_count_max']    = float(wc.max())
    feats['word_count_sum']    = float(wc.sum())
    feats['word_count_std']    = float(wc.std())
    feats['word_count_median'] = float(wc.median())
    feats['word_count_q1']     = float(wc.quantile(0.25))
    feats['word_count_q3']     = float(wc.quantile(0.75))

    return feats


# ── Steps 4–6: Burst and idle features ───────────────────────────────────────

def _burst_sizes(bool_series: pd.Series) -> list:
    sizes, run = [], 0
    for v in bool_series:
        if v:
            run += 1
        elif run:
            sizes.append(run)
            run = 0
    if run:
        sizes.append(run)
    return sizes


def _idle_features(essay: pd.DataFrame) -> dict:
    df = essay.copy()
    df['up_time_lagged'] = df['up_time'].shift(1)
    df['gap_sec'] = ((df['down_time'] - df['up_time_lagged']).abs() / 1000).fillna(0)
    df = df[df['activity'].isin(['Input', 'Remove/Cut'])]
    gaps = df['gap_sec'].dropna()

    return {
        'idle_largest_latency':  float(gaps.max())    if len(gaps) else 0.0,
        'idle_smallest_latency': float(gaps.min())    if len(gaps) else 0.0,
        'idle_median_latency':   float(gaps.median()) if len(gaps) else 0.0,
        'idle_mean':             float(gaps.mean())   if len(gaps) else 0.0,
        'idle_std':              float(gaps.std())    if len(gaps) else 0.0,
        'idle_total':            float(gaps.sum())    if len(gaps) else 0.0,
        'pauses_half_sec':       int(((gaps > 0.5) & (gaps < 1.0)).sum()),
        'pauses_1_sec':          int(((gaps > 1.0) & (gaps < 1.5)).sum()),
        'pauses_1_half_sec':     int(((gaps > 1.5) & (gaps < 2.0)).sum()),
        'pauses_2_sec':          int(((gaps > 2.0) & (gaps < 3.0)).sum()),
        'pauses_3_sec':          int((gaps > 3.0).sum()),
    }


def _p_burst_features(essay: pd.DataFrame) -> dict:
    df = essay.copy()
    df['up_time_lagged'] = df['up_time'].shift(1)
    df['gap_sec'] = ((df['down_time'] - df['up_time_lagged']).abs() / 1000).fillna(0)
    df = df[df['activity'].isin(['Input', 'Remove/Cut'])].copy()
    df['in_burst'] = df['gap_sec'] < 2
    sizes = _burst_sizes(df['in_burst'])
    return {
        'p_burst_count':  len(sizes),
        'p_burst_mean':   float(np.mean(sizes))   if sizes else 0.0,
        'p_burst_std':    float(np.std(sizes))    if sizes else 0.0,
        'p_burst_median': float(np.median(sizes)) if sizes else 0.0,
        'p_burst_max':    float(np.max(sizes))    if sizes else 0.0,
        'p_burst_first':  float(sizes[0])         if sizes else 0.0,
        'p_burst_last':   float(sizes[-1])        if sizes else 0.0,
    }


def _r_burst_features(essay: pd.DataFrame) -> dict:
    df = essay[essay['activity'].isin(['Input', 'Remove/Cut'])].copy()
    df['is_remove'] = df['activity'] == 'Remove/Cut'
    sizes = _burst_sizes(df['is_remove'])
    return {
        'r_burst_mean':   float(np.mean(sizes))   if sizes else 0.0,
        'r_burst_std':    float(np.std(sizes))    if sizes else 0.0,
        'r_burst_median': float(np.median(sizes)) if sizes else 0.0,
        'r_burst_max':    float(np.max(sizes))    if sizes else 0.0,
        'r_burst_first':  float(sizes[0])         if sizes else 0.0,
    }


# ── Step 7: Essay reconstruction ─────────────────────────────────────────────

def _reconstruct_essay(essay: pd.DataFrame) -> str:
    text = ""
    for _, row in essay[['activity', 'cursor_position', 'text_change']].iterrows():
        act = row['activity']
        cur = row['cursor_position']
        tc  = row['text_change']

        if act == 'Replace':
            parts = tc.split(' => ')
            if len(parts) == 2:
                old, new = parts
                text = text[:cur - len(new)] + new + text[cur - len(new) + len(old):]
        elif act == 'Paste':
            text = text[:cur - len(tc)] + tc + text[cur - len(tc):]
        elif act == 'Remove/Cut':
            text = text[:cur] + text[cur + len(tc):]
        elif act.startswith('Move From'):
            try:
                crp  = act[10:]
                spl  = crp.split(' To ')
                vals = [v.split(', ') for v in spl]
                x1, y1 = int(vals[0][0][1:]), int(vals[0][1][:-1])
                x2, y2 = int(vals[1][0][1:]), int(vals[1][1][:-1])
                if x1 != x2:
                    if x1 < x2:
                        text = text[:x1] + text[y1:y2] + text[x1:y1] + text[y2:]
                    else:
                        text = text[:x2] + text[x1:y1] + text[x2:x1] + text[y1:]
            except Exception:
                pass
        else:
            text = text[:cur - len(tc)] + tc + text[cur - len(tc):]

    return text


# ── Steps 8–10: Text statistics ───────────────────────────────────────────────

def _agg(values: list, prefix: str) -> dict:
    if not values:
        names = ['count', 'mean', 'min', 'max', 'first', 'last', 'q1', 'median', 'q3', 'sum']
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


def _word_features(text: str) -> dict:
    raw_words = [w for w in re.split(r'[ \n\.\?\!]', text) if w]
    lengths   = [len(w) for w in raw_words]
    feats     = _agg(lengths, 'word_len')
    feats.pop('word_len_min', None)
    return feats


def _sentence_features(text: str) -> dict:
    raw_sents = [s.replace('\n', '').strip()
                 for s in re.split(r'[\.\?\!]', text)
                 if s.strip()]

    sent_lens = [len(s)         for s in raw_sents]
    sent_wc   = [len(s.split()) for s in raw_sents]

    feats = {'sent_count': len(raw_sents)}

    sent_len_feats = _agg(sent_lens, 'sent_len')
    sent_len_feats.pop('sent_len_count', None)
    feats.update(sent_len_feats)

    sent_wc_feats = _agg(sent_wc, 'sent_word_count')
    sent_wc_feats.pop('sent_word_count_count', None)
    sent_wc_feats.pop('sent_word_count_min', None)
    sent_wc_feats.pop('sent_word_count_sum', None)
    feats.update(sent_wc_feats)

    return feats


def _paragraph_features(text: str) -> dict:
    raw_paras = [p.strip() for p in text.split('\n') if p.strip()]

    para_lens = [len(p)         for p in raw_paras]
    para_wc   = [len(p.split()) for p in raw_paras]

    feats = {'paragraph_count': len(raw_paras)}

    para_len_feats = _agg(para_lens, 'paragraph_len')
    para_len_feats.pop('paragraph_len_count', None)
    feats.update(para_len_feats)

    para_wc_feats = _agg(para_wc, 'paragraph_word_count')
    para_wc_feats.pop('paragraph_word_count_count', None)
    feats.update(para_wc_feats)

    return feats


# ── Step 11: Efficiency features ─────────────────────────────────────────────

def _efficiency_features(essay: pd.DataFrame, essay_text: str) -> dict:
    keys_pressed    = int(essay['activity'].isin(['Input', 'Remove/Cut']).sum())
    session_seconds = (essay['up_time'].max() - essay['down_time'].min()) / 1000
    return {
        'product_to_keys': len(essay_text) / max(keys_pressed, 1),
        'keys_per_second': keys_pressed    / max(session_seconds, 1),
    }


# ── Step 13: Per-word normalisation ──────────────────────────────────────────

def _normalize_counts(row: dict) -> dict:
    word_count = max(row.get('word_len_count', 1), 1)
    return {
        'comma_per_word':            row.get('down_event_comma_cnt', 0)         / word_count,
        'comma_text_per_word':       row.get('text_change_comma_cnt', 0)        / word_count,
        'period_per_word':           row.get('down_event_period_cnt', 0)        / word_count,
        'period_text_per_word':      row.get('text_change_period_cnt', 0)       / word_count,
        'enter_per_word':            row.get('down_event_Enter_cnt', 0)         / word_count,
        'newline_per_word':          row.get('text_change_newline_cnt', 0)      / word_count,
        'shift_per_word':            row.get('down_event_Shift_cnt', 0)         / word_count,
        'capslock_per_word':         row.get('down_event_CapsLock_cnt', 0)      / word_count,
        'dash_per_word':             row.get('text_change_dash_cnt', 0)         / word_count,
        'question_per_word':         row.get('text_change_question_cnt', 0)     / word_count,
        'quote_per_word':            row.get('text_change_quote_cnt', 0)        / word_count,
        'q_keys_per_word':           row.get('down_event_q_cnt', 0)             / word_count,
        'q_text_per_word':           row.get('text_change_q_cnt', 0)            / word_count,
        'space_per_word':            row.get('down_event_Space_cnt', 0)         / word_count,
        'space_text_per_word':       row.get('text_change_space_cnt', 0)        / word_count,
        'pauses_half_per_word':      row.get('pauses_half_sec', 0)              / word_count,
        'pauses_1_per_word':         row.get('pauses_1_sec', 0)                 / word_count,
        'pauses_1half_per_word':     row.get('pauses_1_half_sec', 0)            / word_count,
        'pauses_2_per_word':         row.get('pauses_2_sec', 0)                 / word_count,
        'pauses_3_per_word':         row.get('pauses_3_sec', 0)                 / word_count,
        'removecut_per_word':        row.get('activity_RemoveCut_cnt', 0)       / word_count,
        'backspace_per_word':        row.get('down_event_Backspace_cnt', 0)     / word_count,
        'input_per_word':            row.get('activity_Input_cnt', 0)           / word_count,
        'nonproduction_per_word':    row.get('activity_Nonproduction_cnt', 0)   / word_count,
        'leftclick_per_word':        row.get('down_event_Leftclick_cnt', 0)     / word_count,
        'arrowleft_per_word':        row.get('down_event_ArrowLeft_cnt', 0)     / word_count,
        'arrowright_per_word':       row.get('down_event_ArrowRight_cnt', 0)    / word_count,
        'sent_per_word':             row.get('sent_count', 0)                   / word_count,
        'p_burst_per_word':          row.get('p_burst_count', 0)                / word_count,
        'replace_per_word':          row.get('activity_Replace_cnt', 0)         / word_count,
        'paste_per_word':            row.get('activity_Paste_cnt', 0)           / word_count,
        'doublequote_text_per_word': row.get('text_change_doublequote_cnt', 0)  / word_count,
        'semicolon_text_per_word':   row.get('text_change_semicolon_cnt', 0)    / word_count,
        'equals_text_per_word':      row.get('text_change_equals_cnt', 0)       / word_count,
        'slash_text_per_word':       row.get('text_change_slash_cnt', 0)        / word_count,
        'backslash_text_per_word':   row.get('text_change_backslash_cnt', 0)    / word_count,
        'colon_text_per_word':       row.get('text_change_colon_cnt', 0)        / word_count,
        'arrowdown_per_word':        row.get('down_event_ArrowDown_cnt', 0)     / word_count,
        'arrowup_per_word':          row.get('down_event_ArrowUp_cnt', 0)       / word_count,
        'singlequote_per_word':      row.get('down_event_singlequote_cnt', 0)   / word_count,
        'delete_per_word':           row.get('down_event_Delete_cnt', 0)        / word_count,
        'unidentified_per_word':     row.get('down_event_Unidentified_cnt', 0)  / word_count,
    }


# ── Step 15: Readability & structural consistency ─────────────────────────────

def _readability_features(text: str, row: dict) -> dict:
    chars  = len(re.sub(r'\s', '', text))
    words  = max(row.get('word_len_count', 1), 1)
    sents  = max(row.get('sent_count', 1), 1)

    ari = 4.71 * (chars / words) + 0.5 * (words / sents) - 21.43

    L   = (chars / words) * 100
    S   = (sents / words) * 100
    cli = 0.0588 * L - 0.296 * S - 15.8

    raw_sents = [s.strip() for s in re.split(r'[\.\?\!]', text) if s.strip()]
    sent_lens = np.array([len(s) for s in raw_sents], dtype=float)
    sent_std  = float(sent_lens.std())  if len(sent_lens) > 1 else 0.0
    sent_cv   = float(sent_std / (sent_lens.mean() + 1e-9))

    raw_paras = [p.strip() for p in text.split('\n') if p.strip()]
    para_lens = np.array([len(p) for p in raw_paras], dtype=float)
    para_std  = float(para_lens.std())  if len(para_lens) > 1 else 0.0
    para_cv   = float(para_std / (para_lens.mean() + 1e-9))

    para_wc  = np.array([len(p.split()) for p in raw_paras], dtype=float)
    para_bal = float(para_wc.std()) if len(para_wc) > 1 else 0.0

    dur_min = max(row.get('session_duration_sec', 1), 1) / 60.0
    wpm     = words / dur_min

    return {
        'ari_score':          float(ari),
        'coleman_liau_score': float(cli),
        'sent_len_std':       sent_std,
        'sent_len_cv':        sent_cv,
        'para_len_std':       para_std,
        'para_len_cv':        para_cv,
        'para_balance':       para_bal,
        'words_per_minute':   float(wpm),
    }


# ── Step 16: Revision timing ─────────────────────────────────────────────────

def _revision_timing_features(essay: pd.DataFrame) -> dict:
    t_min = essay['down_time'].min()
    t_max = essay['down_time'].max()
    t_dur = max(t_max - t_min, 1)

    remove_events = essay[essay['activity'] == 'Remove/Cut']['down_time']

    if remove_events.empty:
        return {
            'revision_ratio_early': 0.0,
            'revision_ratio_mid':   0.0,
            'revision_ratio_late':  0.0,
            'revision_timing_mean': 0.0,
            'revision_timing_std':  0.0,
        }

    norm_times = (remove_events - t_min) / t_dur
    total      = len(norm_times)

    return {
        'revision_ratio_early': float((norm_times < 1/3).sum())                         / total,
        'revision_ratio_mid':   float(((norm_times >= 1/3) & (norm_times < 2/3)).sum()) / total,
        'revision_ratio_late':  float((norm_times >= 2/3).sum())                        / total,
        'revision_timing_mean': float(norm_times.mean()),
        'revision_timing_std':  float(norm_times.std()) if total > 1 else 0.0,
    }


# ── Main pipeline ─────────────────────────────────────────────────────────────

_REPLACE_MAP = {'.': 'period', ',': 'comma', "'": 'singlequote'}

def compute_features(
    logs: pd.DataFrame,
    tfidf_pipeline,
    event_tfidf_pipeline,
) -> pd.DataFrame:
    """
    Compute v6 per-essay features.

    Parameters
    ----------
    logs : pd.DataFrame        — raw keystroke logs
    tfidf_pipeline             — fitted text TF-IDF Pipeline (loaded from TFIDF_PATH)
    event_tfidf_pipeline       — fitted event TF-IDF Pipeline (loaded from EVENT_TFIDF_PATH)

    Returns
    -------
    pd.DataFrame — one row per essay, columns = ['id'] + FEATURE_COLS
    """
    logs = logs.sort_values(['id', 'event_id']).reset_index(drop=True)

    rows        = []
    texts       = {}
    event_seqs  = {}

    for essay_id, essay in logs.groupby('id'):
        essay = essay.reset_index(drop=True)
        row   = {'id': essay_id}

        row.update(_count_features(essay))
        row.update(_input_word_features(essay))
        row.update(_timing_features(essay))
        row.update(_idle_features(essay))
        row.update(_p_burst_features(essay))
        row.update(_r_burst_features(essay))

        non_nonproduction = essay[essay['activity'] != 'Nonproduction']
        text = _reconstruct_essay(non_nonproduction)
        texts[essay_id] = text

        event_seqs[essay_id] = ' '.join(
            _REPLACE_MAP.get(e, e) for e in essay['down_event'].astype(str)
        )

        row.update(_word_features(text))
        row.update(_sentence_features(text))
        row.update(_paragraph_features(text))
        row.update(_efficiency_features(essay, text))

        row['session_duration_sec'] = (
            row.get('down_time_max', 0) - row.get('down_time_min', 0)
        ) / 1000

        row.update(_normalize_counts(row))
        row.update(_readability_features(text, row))
        row.update(_revision_timing_features(essay))

        rows.append(row)

    df = pd.DataFrame(rows).fillna(0)

    essay_ids    = df['id'].tolist()
    corpus       = [texts[eid]      for eid in essay_ids]
    event_corpus = [event_seqs[eid] for eid in essay_ids]

    # TF-IDF 1: char n-gram on reconstructed text (transform only)
    svd_matrix = tfidf_pipeline.transform(corpus)
    svd_df     = pd.DataFrame(svd_matrix, columns=TFIDF_SVD_COLS)
    svd_df['id'] = df['id'].values
    df = df.merge(svd_df, on='id', how='left')

    # TF-IDF 2: word n-gram on down_event sequence (transform only)
    event_svd_matrix = event_tfidf_pipeline.transform(event_corpus)
    event_svd_df     = pd.DataFrame(event_svd_matrix, columns=EVENT_TFIDF_SVD_COLS)
    event_svd_df['id'] = df['id'].values
    df = df.merge(event_svd_df, on='id', how='left')

    return df[['id'] + FEATURE_COLS].fillna(0)


# =============================================================================
# SECTION 1 – LOAD TF-IDF PIPELINES
# =============================================================================
print(f"Loading text TF-IDF SVD pipeline from {TFIDF_PATH}...")
with open(TFIDF_PATH, 'rb') as f:
    tfidf_pipeline = pickle.load(f)

print(f"Loading event TF-IDF SVD pipeline from {EVENT_TFIDF_PATH}...")
with open(EVENT_TFIDF_PATH, 'rb') as f:
    event_tfidf_pipeline = pickle.load(f)

# =============================================================================
# SECTION 2 – LOAD LGBM MODELS
# =============================================================================
model_paths = sorted(glob.glob(f'{MODELS_DIR}/lgbm_s*_fold*.txt'))
if not model_paths:
    raise FileNotFoundError(
        f"No fold models found in {MODELS_DIR}/\n"
        "Upload lgbm_s*_fold*.txt models to the dataset."
    )

models = [lgb.Booster(model_file=p) for p in model_paths]
print(f"Loaded {len(models)} LightGBM models from {MODELS_DIR}/")

# =============================================================================
# SECTION 3 – FEATURE ENGINEERING
# =============================================================================
print("Loading test_logs.csv...")
logs = pd.read_csv(f'{DATA_DIR}/test_logs.csv')
print(f"  {logs.shape[0]:,} rows | {logs['id'].nunique()} essays")

print("Computing per-essay features (v6)...")
test_df = compute_features(logs, tfidf_pipeline, event_tfidf_pipeline).fillna(0)
X_test  = test_df[FEATURE_COLS].values
print(f"  Feature matrix: {X_test.shape[0]} essays × {X_test.shape[1]} features")

# =============================================================================
# SECTION 4 – INFERENCE
# =============================================================================
print("\nRunning inference...")
fold_preds = np.stack([model.predict(X_test) for model in models], axis=0)
preds      = fold_preds.mean(axis=0)
preds      = np.clip(preds, 0.5, 6.0)

print(f"  Prediction stats — mean: {preds.mean():.3f} | "
      f"std: {preds.std():.3f} | min: {preds.min():.3f} | max: {preds.max():.3f}")

# =============================================================================
# SECTION 5 – WRITE SUBMISSION
# =============================================================================
submission = pd.DataFrame({'id': test_df['id'], 'score': preds})
submission.to_csv(OUTPUT_PATH, index=False)
print(f"\nSubmission saved to: {OUTPUT_PATH}")
print(f"  {len(submission)} rows")
print(submission.head(10).to_string(index=False))
