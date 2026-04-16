"""
v6 Features: Linking Writing Process to Writing Quality
========================================================
Base: v5_features (206 features)

New features added in v6:
  Event-sequence TF-IDF SVD (20 components) — v6 new
    Each essay's down_event sequence is joined as a space-separated string,
    then vectorised with TfidfVectorizer(analyzer='word', ngram_range=(1, 5))
    and compressed with TruncatedSVD(20).
    Captures rhythmic keystroke patterns (e.g. common Shift-q runs,
    Backspace clusters, ArrowKey navigation sequences).

Total: 206 (v5) + 20 new = 226 features

Two TF-IDF pipelines (both must be saved/loaded at inference time):
  tfidf_pipeline       — char 2–4-gram on reconstructed essay text (same as v4/v5)
  event_tfidf_pipeline — word 1–5-gram on down_event sequence (new in v6)

Usage
-----
    # Training
    from v6_features import compute_features, FEATURE_COLS
    train_df, tfidf_pipeline, event_tfidf_pipeline = compute_features(train_logs)

    # Test / inference
    test_df, _, _ = compute_features(
        test_logs,
        tfidf_pipeline=tfidf_pipeline,
        event_tfidf_pipeline=event_tfidf_pipeline,
    )
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import re
import numpy as np
import pandas as pd
from scipy.stats import skew as _skew
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

# Unchanged from v4 — import directly
from preprocess.essay_reconstruction import reconstruct_essay
from preprocess.burst_stats          import idle_features, p_burst_features, r_burst_features
from preprocess.text_stats           import word_features, sentence_features, paragraph_features
from preprocess.efficiency           import efficiency_features

# ── TF-IDF configs ────────────────────────────────────────────────────────────
TFIDF_N_COMPONENTS       = 20
TFIDF_SVD_COLS           = [f'tfidf_svd_{i}'       for i in range(TFIDF_N_COMPONENTS)]

EVENT_TFIDF_N_COMPONENTS = 20
EVENT_TFIDF_SVD_COLS     = [f'event_tfidf_svd_{i}' for i in range(EVENT_TFIDF_N_COMPONENTS)]


# =============================================================================
# STEP 1 — COUNT FEATURES  (identical to v5)
# =============================================================================

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


# =============================================================================
# STEP 2 — INPUT WORD FEATURES  (identical to v5)
# =============================================================================

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


# =============================================================================
# STEP 3 — TIMING FEATURES  (identical to v5)
# =============================================================================

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


# =============================================================================
# STEP 13 — PER-WORD NORMALISATION  (identical to v5)
# =============================================================================

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


# =============================================================================
# STEPS 14–15 — READABILITY + REVISION TIMING  (identical to v5)
# =============================================================================

def _readability_features(text: str, row: dict) -> dict:
    chars = len(re.sub(r'\s', '', text))
    words = max(row.get('word_len_count', 1), 1)
    sents = max(row.get('sent_count', 1), 1)

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


# =============================================================================
# COMPUTE FEATURES  — two TF-IDF pipelines
# =============================================================================

def compute_features(
    logs: pd.DataFrame,
    tfidf_pipeline=None,
    event_tfidf_pipeline=None,
):
    """
    Compute per-essay v6 features from raw keystroke logs.

    Parameters
    ----------
    logs : pd.DataFrame
        Raw keystroke logs.
    tfidf_pipeline : fitted sklearn Pipeline or None
        Text TF-IDF (char 2–4-gram on reconstructed essay).
        None → fit on corpus (training mode).
    event_tfidf_pipeline : fitted sklearn Pipeline or None
        Event TF-IDF (word 1–5-gram on down_event sequence).
        None → fit on corpus (training mode).

    Returns
    -------
    tuple (pd.DataFrame, tfidf_pipeline, event_tfidf_pipeline)
        DataFrame has columns ['id'] + FEATURE_COLS (226 features).
    """
    logs = logs.sort_values(['id', 'event_id']).reset_index(drop=True)

    rows         = []
    texts        = {}
    event_seqs   = {}   # down_event sequence per essay (space-joined)

    for essay_id, essay in logs.groupby('id'):
        essay = essay.reset_index(drop=True)
        row   = {'id': essay_id}

        row.update(_count_features(essay))
        row.update(_input_word_features(essay))
        row.update(_timing_features(essay))
        row.update(idle_features(essay))
        row.update(p_burst_features(essay))
        row.update(r_burst_features(essay))

        non_nonproduction = essay[essay['activity'] != 'Nonproduction']
        text = reconstruct_essay(non_nonproduction)
        texts[essay_id] = text

        # Build the event sequence document: one token per down_event row
        # Replace punctuation tokens that could confuse the word tokeniser
        # (period, comma, single-quote) with safe placeholder strings.
        _REPLACE_MAP = {'.': 'period', ',': 'comma', "'": 'singlequote'}
        event_seq = ' '.join(
            _REPLACE_MAP.get(e, e) for e in essay['down_event'].astype(str)
        )
        event_seqs[essay_id] = event_seq

        row.update(word_features(text))
        row.update(sentence_features(text))
        row.update(paragraph_features(text))
        row.update(efficiency_features(essay, text))

        row['session_duration_sec'] = (
            row.get('down_time_max', 0) - row.get('down_time_min', 0)
        ) / 1000

        row.update(_normalize_counts(row))
        row.update(_readability_features(text, row))
        row.update(_revision_timing_features(essay))

        rows.append(row)

    df = pd.DataFrame(rows).fillna(0)

    essay_ids   = df['id'].tolist()
    corpus      = [texts[eid]      for eid in essay_ids]
    event_corpus= [event_seqs[eid] for eid in essay_ids]

    # ── TF-IDF 1: char n-gram on reconstructed text (same as v4/v5) ──────────
    if tfidf_pipeline is None:
        tfidf_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                analyzer    = 'char_wb',
                ngram_range = (2, 4),
                max_features= 500,
                sublinear_tf= True,
            )),
            ('svd', TruncatedSVD(n_components=TFIDF_N_COMPONENTS, random_state=42)),
        ])
        svd_matrix = tfidf_pipeline.fit_transform(corpus)
    else:
        svd_matrix = tfidf_pipeline.transform(corpus)

    svd_df       = pd.DataFrame(svd_matrix, columns=TFIDF_SVD_COLS)
    svd_df['id'] = df['id'].values
    df = df.merge(svd_df, on='id', how='left')

    # ── TF-IDF 2: word n-gram on down_event sequence (v6 new) ─────────────────
    if event_tfidf_pipeline is None:
        event_tfidf_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                analyzer    = 'word',
                ngram_range = (1, 5),
                sublinear_tf= True,
            )),
            ('svd', TruncatedSVD(n_components=EVENT_TFIDF_N_COMPONENTS, random_state=42)),
        ])
        event_svd_matrix = event_tfidf_pipeline.fit_transform(event_corpus)
    else:
        event_svd_matrix = event_tfidf_pipeline.transform(event_corpus)

    event_svd_df       = pd.DataFrame(event_svd_matrix, columns=EVENT_TFIDF_SVD_COLS)
    event_svd_df['id'] = df['id'].values
    df = df.merge(event_svd_df, on='id', how='left')

    return df[['id'] + FEATURE_COLS].fillna(0), tfidf_pipeline, event_tfidf_pipeline


# =============================================================================
# FEATURE COLUMNS  (v5 206 + 20 event TF-IDF = 226 features)
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

    # ── 13. TEXT TF-IDF SVD  (char 2–4-gram, same as v4/v5) ─────────────────
    *TFIDF_SVD_COLS,         # tfidf_svd_0 … tfidf_svd_19

    # ── 14. EVENT TF-IDF SVD  (word 1–5-gram on down_event sequence, v6 new) ─
    *EVENT_TFIDF_SVD_COLS,   # event_tfidf_svd_0 … event_tfidf_svd_19
]


# =============================================================================
# CATEGORIES
# =============================================================================

CATEGORIES = {
    # 1. Word Characteristics
    'word_len_sum':               'Word Characteristics',
    'word_len_mean':              'Word Characteristics',
    'word_len_q3':                'Word Characteristics',
    'word_len_max':               'Word Characteristics',
    'word_len_first':             'Word Characteristics',
    'word_len_last':              'Word Characteristics',
    'word_len_count':             'Word Characteristics',
    'input_word_count':           'Word Characteristics',
    'input_word_length_mean':     'Word Characteristics',
    'input_word_length_max':      'Word Characteristics',
    'input_word_length_std':      'Word Characteristics',
    'input_word_length_median':   'Word Characteristics',
    'input_word_length_skew':     'Word Characteristics',
    'word_count_min':             'Word Characteristics',
    'word_count_mean':            'Word Characteristics',
    'word_count_max':             'Word Characteristics',
    'word_count_sum':             'Word Characteristics',
    'word_count_std':             'Word Characteristics',
    'word_count_median':          'Word Characteristics',
    'word_count_q1':              'Word Characteristics',
    'word_count_q3':              'Word Characteristics',

    # 2. Sentence Characteristics
    'sent_len_mean':              'Sentence Characteristics',
    'sent_len_median':            'Sentence Characteristics',
    'sent_len_min':               'Sentence Characteristics',
    'sent_len_max':               'Sentence Characteristics',
    'sent_len_first':             'Sentence Characteristics',
    'sent_len_last':              'Sentence Characteristics',
    'sent_len_q1':                'Sentence Characteristics',
    'sent_len_q3':                'Sentence Characteristics',
    'sent_per_word':              'Sentence Characteristics',
    'sent_word_count_mean':       'Sentence Characteristics',
    'sent_word_count_max':        'Sentence Characteristics',
    'sent_word_count_first':      'Sentence Characteristics',
    'sent_word_count_last':       'Sentence Characteristics',
    'sent_word_count_q1':         'Sentence Characteristics',
    'sent_word_count_median':     'Sentence Characteristics',
    'sent_word_count_q3':         'Sentence Characteristics',

    # 3. Paragraph Characteristics
    'paragraph_len_mean':              'Paragraph Characteristics',
    'paragraph_len_median':            'Paragraph Characteristics',
    'paragraph_len_min':               'Paragraph Characteristics',
    'paragraph_len_max':               'Paragraph Characteristics',
    'paragraph_len_first':             'Paragraph Characteristics',
    'paragraph_len_last':              'Paragraph Characteristics',
    'paragraph_len_q1':                'Paragraph Characteristics',
    'paragraph_len_q3':                'Paragraph Characteristics',
    'paragraph_count':                 'Paragraph Characteristics',
    'paragraph_word_count_mean':       'Paragraph Characteristics',
    'paragraph_word_count_min':        'Paragraph Characteristics',
    'paragraph_word_count_max':        'Paragraph Characteristics',
    'paragraph_word_count_first':      'Paragraph Characteristics',
    'paragraph_word_count_last':       'Paragraph Characteristics',
    'paragraph_word_count_q1':         'Paragraph Characteristics',
    'paragraph_word_count_median':     'Paragraph Characteristics',
    'paragraph_word_count_q3':         'Paragraph Characteristics',

    # 4. Punctuation & Formatting
    'comma_per_word':             'Punctuation & Formatting',
    'comma_text_per_word':        'Punctuation & Formatting',
    'period_per_word':            'Punctuation & Formatting',
    'period_text_per_word':       'Punctuation & Formatting',
    'enter_per_word':             'Punctuation & Formatting',
    'newline_per_word':           'Punctuation & Formatting',
    'shift_per_word':             'Punctuation & Formatting',
    'capslock_per_word':          'Punctuation & Formatting',
    'dash_per_word':              'Punctuation & Formatting',
    'question_per_word':          'Punctuation & Formatting',
    'quote_per_word':             'Punctuation & Formatting',
    'doublequote_text_per_word':  'Punctuation & Formatting',
    'semicolon_text_per_word':    'Punctuation & Formatting',
    'equals_text_per_word':       'Punctuation & Formatting',
    'slash_text_per_word':        'Punctuation & Formatting',
    'backslash_text_per_word':    'Punctuation & Formatting',
    'colon_text_per_word':        'Punctuation & Formatting',
    'singlequote_per_word':       'Punctuation & Formatting',

    # 5. Typing Speed & Fluency
    'keys_per_second':            'Typing Speed & Fluency',
    'action_time_min':            'Typing Speed & Fluency',
    'action_time_sum':            'Typing Speed & Fluency',
    'action_time_mean':           'Typing Speed & Fluency',
    'action_time_std':            'Typing Speed & Fluency',
    'action_time_median':         'Typing Speed & Fluency',
    'action_time_max':            'Typing Speed & Fluency',
    'action_time_q1':             'Typing Speed & Fluency',
    'action_time_q3':             'Typing Speed & Fluency',
    'q_keys_per_word':            'Typing Speed & Fluency',
    'q_text_per_word':            'Typing Speed & Fluency',
    'space_per_word':             'Typing Speed & Fluency',
    'space_text_per_word':        'Typing Speed & Fluency',
    'n_unique_activity':          'Typing Speed & Fluency',
    'n_unique_text_change':       'Typing Speed & Fluency',
    'n_unique_down_event':        'Typing Speed & Fluency',
    'n_unique_up_event':          'Typing Speed & Fluency',

    # 6. Pausing Behavior
    'idle_largest_latency':       'Pausing Behavior',
    'idle_median_latency':        'Pausing Behavior',
    'idle_mean':                  'Pausing Behavior',
    'idle_std':                   'Pausing Behavior',
    'idle_total':                 'Pausing Behavior',
    'pauses_half_per_word':       'Pausing Behavior',
    'pauses_1_per_word':          'Pausing Behavior',
    'pauses_1half_per_word':      'Pausing Behavior',
    'pauses_2_per_word':          'Pausing Behavior',
    'pauses_3_per_word':          'Pausing Behavior',

    # 7. Revision Behavior
    'removecut_per_word':         'Revision Behavior',
    'backspace_per_word':         'Revision Behavior',
    'delete_per_word':            'Revision Behavior',
    'r_burst_mean':               'Revision Behavior',
    'r_burst_std':                'Revision Behavior',
    'r_burst_median':             'Revision Behavior',
    'r_burst_max':                'Revision Behavior',
    'r_burst_first':              'Revision Behavior',

    # 8. Production Flow & Navigation
    'input_per_word':             'Production Flow & Navigation',
    'replace_per_word':           'Production Flow & Navigation',
    'paste_per_word':             'Production Flow & Navigation',
    'nonproduction_per_word':     'Production Flow & Navigation',
    'p_burst_per_word':           'Production Flow & Navigation',
    'p_burst_mean':               'Production Flow & Navigation',
    'p_burst_std':                'Production Flow & Navigation',
    'p_burst_median':             'Production Flow & Navigation',
    'p_burst_max':                'Production Flow & Navigation',
    'p_burst_first':              'Production Flow & Navigation',
    'p_burst_last':               'Production Flow & Navigation',
    'product_to_keys':            'Production Flow & Navigation',
    'session_duration_sec':       'Production Flow & Navigation',
    'cursor_position_min':        'Production Flow & Navigation',
    'cursor_position_mean':       'Production Flow & Navigation',
    'cursor_position_std':        'Production Flow & Navigation',
    'cursor_position_median':     'Production Flow & Navigation',
    'cursor_position_q1':         'Production Flow & Navigation',
    'cursor_position_q3':         'Production Flow & Navigation',
    'cursor_position_max':        'Production Flow & Navigation',
    'leftclick_per_word':         'Production Flow & Navigation',
    'arrowleft_per_word':         'Production Flow & Navigation',
    'arrowright_per_word':        'Production Flow & Navigation',
    'arrowdown_per_word':         'Production Flow & Navigation',
    'arrowup_per_word':           'Production Flow & Navigation',
    'unidentified_per_word':      'Production Flow & Navigation',
    'down_time_min':              'Production Flow & Navigation',
    'down_time_mean':             'Production Flow & Navigation',
    'down_time_std':              'Production Flow & Navigation',
    'down_time_median':           'Production Flow & Navigation',
    'down_time_q1':               'Production Flow & Navigation',
    'down_time_q3':               'Production Flow & Navigation',
    'down_time_max':              'Production Flow & Navigation',
    'up_time_min':                'Production Flow & Navigation',
    'up_time_mean':               'Production Flow & Navigation',
    'up_time_std':                'Production Flow & Navigation',
    'up_time_median':             'Production Flow & Navigation',
    'up_time_q1':                 'Production Flow & Navigation',
    'up_time_q3':                 'Production Flow & Navigation',
    'up_time_max':                'Production Flow & Navigation',

    # 9. Up Event Counts
    'up_event_q_cnt':             'Up Event Counts',
    'up_event_Space_cnt':         'Up Event Counts',
    'up_event_Backspace_cnt':     'Up Event Counts',
    'up_event_Shift_cnt':         'Up Event Counts',
    'up_event_ArrowRight_cnt':    'Up Event Counts',
    'up_event_Leftclick_cnt':     'Up Event Counts',
    'up_event_ArrowLeft_cnt':     'Up Event Counts',
    'up_event_period_cnt':        'Up Event Counts',
    'up_event_comma_cnt':         'Up Event Counts',
    'up_event_ArrowDown_cnt':     'Up Event Counts',
    'up_event_ArrowUp_cnt':       'Up Event Counts',
    'up_event_Enter_cnt':         'Up Event Counts',
    'up_event_CapsLock_cnt':      'Up Event Counts',
    'up_event_singlequote_cnt':   'Up Event Counts',
    'up_event_Delete_cnt':        'Up Event Counts',
    'up_event_Unidentified_cnt':  'Up Event Counts',

    # 10. Raw Event Counts
    'activity_Replace_cnt':       'Raw Event Counts',
    'activity_Paste_cnt':         'Raw Event Counts',
    'text_change_doublequote_cnt':'Raw Event Counts',
    'text_change_semicolon_cnt':  'Raw Event Counts',
    'text_change_equals_cnt':     'Raw Event Counts',
    'text_change_slash_cnt':      'Raw Event Counts',
    'text_change_backslash_cnt':  'Raw Event Counts',
    'text_change_colon_cnt':      'Raw Event Counts',
    'down_event_ArrowDown_cnt':   'Raw Event Counts',
    'down_event_ArrowUp_cnt':     'Raw Event Counts',
    'down_event_singlequote_cnt': 'Raw Event Counts',
    'down_event_Delete_cnt':      'Raw Event Counts',
    'down_event_Unidentified_cnt':'Raw Event Counts',

    # 11. Readability & Structural Consistency
    'ari_score':                  'Readability & Consistency',
    'coleman_liau_score':         'Readability & Consistency',
    'sent_len_std':               'Readability & Consistency',
    'sent_len_cv':                'Readability & Consistency',
    'para_len_std':               'Readability & Consistency',
    'para_len_cv':                'Readability & Consistency',
    'para_balance':               'Readability & Consistency',
    'words_per_minute':           'Readability & Consistency',

    # 12. Revision Timing
    'revision_ratio_early':       'Revision Timing',
    'revision_ratio_mid':         'Revision Timing',
    'revision_ratio_late':        'Revision Timing',
    'revision_timing_mean':       'Revision Timing',
    'revision_timing_std':        'Revision Timing',

    # 13. Text TF-IDF SVD
    **{col: 'TF-IDF SVD (Text)'  for col in TFIDF_SVD_COLS},

    # 14. Event TF-IDF SVD (v6 new)
    **{col: 'TF-IDF SVD (Events)' for col in EVENT_TFIDF_SVD_COLS},
}


# =============================================================================
# COLOUR PALETTE
# =============================================================================

CAT_PALETTE = {
    'Word Characteristics':         '#4C72B0',
    'Sentence Characteristics':     '#DD8452',
    'Paragraph Characteristics':    '#55A868',
    'Punctuation & Formatting':     '#8172B2',
    'Typing Speed & Fluency':       '#C44E52',
    'Pausing Behavior':             '#937860',
    'Revision Behavior':            '#DA8BC3',
    'Production Flow & Navigation': '#8C8C8C',
    'Up Event Counts':              '#E8A838',
    'Raw Event Counts':             '#6DAEDB',
    'Readability & Consistency':    '#E377C2',
    'Revision Timing':              '#17BECF',
    'TF-IDF SVD (Text)':            '#BCBD22',
    'TF-IDF SVD (Events)':          '#9467BD',   # v6 new
}
