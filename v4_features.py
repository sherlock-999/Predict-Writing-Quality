"""
v4 Features: Linking Writing Process to Writing Quality
========================================================
Base: v2_features (119 features, per-word normalised counts)

New features added in v4:
  Category 9  — Readability & Structural Consistency (8 features)
    ari_score             — Automated Readability Index (char/word/sentence ratio)
    coleman_liau_score    — Coleman-Liau readability index
    sent_len_std          — std of sentence lengths (structural consistency)
    sent_len_cv           — coefficient of variation of sentence lengths
    para_len_std          — std of paragraph lengths
    para_len_cv           — coefficient of variation of paragraph lengths
    para_balance          — std of paragraph word counts (balance of development)
    words_per_minute      — gross writing speed

  Category 10 — Revision Timing (5 features)
    revision_ratio_early  — share of Remove/Cut events in first third of session
    revision_ratio_mid    — share of Remove/Cut events in middle third
    revision_ratio_late   — share of Remove/Cut events in last third
    revision_timing_mean  — mean normalised time of Remove/Cut events (0=start, 1=end)
    revision_timing_std   — std of normalised revision times

  Category 11 — TF-IDF SVD (20 features)
    tfidf_svd_0 … tfidf_svd_19
      — TruncatedSVD of char-bigram TF-IDF on reconstructed text.
         Since letters are anonymised as 'q', bigrams capture punctuation
         rhythm and sentence-break patterns rather than vocabulary.
         Fitted on the full corpus passed to compute_features().

Total: 119 (v2) + 8 + 5 + 20 = 152 features

Pipeline stages
---------------
  Steps 1–13  same as v2_features (count → normalize → session_duration)
  Step  14    readability + consistency + pace
  Step  15    revision timing
  Step  16    TF-IDF SVD (corpus-level, two-pass)

Two-pass design
---------------
  compute_features() accepts an optional `tfidf_pipeline` argument.
  • If None  → a new sklearn Pipeline (TF-IDF + TruncatedSVD) is *fitted*
               on the corpus and returned alongside the feature DataFrame.
               Use this for training.
  • If provided → the pipeline is applied as *transform* only (no refitting).
               Use this for test/inference to ensure the same SVD components
               as training.

  Return value: tuple (DataFrame, fitted_pipeline)

Usage
-----
    # Training
    from v4_features import compute_features, FEATURE_COLS
    train_df, tfidf_pipeline = compute_features(train_logs)

    # Test / inference
    test_df, _ = compute_features(test_logs, tfidf_pipeline=tfidf_pipeline)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

from preprocess.essay_reconstruction import reconstruct_essay
from preprocess.keystroke_stats      import count_features, input_word_features, timing_features
from preprocess.burst_stats          import idle_features, p_burst_features, r_burst_features
from preprocess.text_stats           import word_features, sentence_features, paragraph_features
from preprocess.efficiency           import efficiency_features

# ── TF-IDF config ─────────────────────────────────────────────────────────────
TFIDF_N_COMPONENTS = 20
TFIDF_SVD_COLS     = [f'tfidf_svd_{i}' for i in range(TFIDF_N_COMPONENTS)]


# =============================================================================
# PER-WORD NORMALISATION  (same as v2 — no verbosity_per_word)
# =============================================================================

def _normalize_counts(row: dict) -> dict:
    word_count = max(row.get('word_len_count', 1), 1)
    return {
        'comma_per_word':        row.get('down_event_comma_cnt', 0)       / word_count,
        'comma_text_per_word':   row.get('text_change_comma_cnt', 0)      / word_count,
        'period_per_word':       row.get('down_event_period_cnt', 0)      / word_count,
        'period_text_per_word':  row.get('text_change_period_cnt', 0)     / word_count,
        'enter_per_word':        row.get('down_event_Enter_cnt', 0)       / word_count,
        'newline_per_word':      row.get('text_change_newline_cnt', 0)    / word_count,
        'shift_per_word':        row.get('down_event_Shift_cnt', 0)       / word_count,
        'capslock_per_word':     row.get('down_event_CapsLock_cnt', 0)    / word_count,
        'dash_per_word':         row.get('text_change_dash_cnt', 0)       / word_count,
        'question_per_word':     row.get('text_change_question_cnt', 0)   / word_count,
        'quote_per_word':        row.get('text_change_quote_cnt', 0)      / word_count,
        'q_keys_per_word':       row.get('down_event_q_cnt', 0)           / word_count,
        'q_text_per_word':       row.get('text_change_q_cnt', 0)          / word_count,
        'space_per_word':        row.get('down_event_Space_cnt', 0)       / word_count,
        'space_text_per_word':   row.get('text_change_space_cnt', 0)      / word_count,
        'pauses_half_per_word':  row.get('pauses_half_sec', 0)            / word_count,
        'pauses_1_per_word':     row.get('pauses_1_sec', 0)               / word_count,
        'pauses_1half_per_word': row.get('pauses_1_half_sec', 0)          / word_count,
        'pauses_2_per_word':     row.get('pauses_2_sec', 0)               / word_count,
        'pauses_3_per_word':     row.get('pauses_3_sec', 0)               / word_count,
        'removecut_per_word':    row.get('activity_RemoveCut_cnt', 0)     / word_count,
        'backspace_per_word':    row.get('down_event_Backspace_cnt', 0)   / word_count,
        'input_per_word':        row.get('activity_Input_cnt', 0)         / word_count,
        'nonproduction_per_word':row.get('activity_Nonproduction_cnt', 0) / word_count,
        'leftclick_per_word':    row.get('down_event_Leftclick_cnt', 0)   / word_count,
        'arrowleft_per_word':    row.get('down_event_ArrowLeft_cnt', 0)   / word_count,
        'arrowright_per_word':   row.get('down_event_ArrowRight_cnt', 0)  / word_count,
        'sent_per_word':         row.get('sent_count', 0)                 / word_count,
        'p_burst_per_word':      row.get('p_burst_count', 0)              / word_count,
    }


# =============================================================================
# NEW FEATURES (v4 additions)
# =============================================================================

def _readability_features(text: str, row: dict) -> dict:
    """
    Readability and structural consistency features.
    ARI and Coleman-Liau use only char/word/sentence counts so they work
    correctly even though all letters are anonymised as 'q'.
    """
    chars = len(re.sub(r'\s', '', text))           # non-whitespace characters
    words = max(row.get('word_len_count', 1), 1)
    sents = max(row.get('sent_count', 1), 1)

    # Automated Readability Index
    ari = 4.71 * (chars / words) + 0.5 * (words / sents) - 21.43

    # Coleman-Liau: L = avg chars per 100 words, S = avg sentences per 100 words
    L   = (chars / words) * 100
    S   = (sents / words) * 100
    cli = 0.0588 * L - 0.296 * S - 15.8

    # Sentence length consistency
    raw_sents = [s.strip() for s in re.split(r'[\.\?\!]', text) if s.strip()]
    sent_lens = np.array([len(s) for s in raw_sents], dtype=float)
    sent_std  = float(sent_lens.std())  if len(sent_lens) > 1 else 0.0
    sent_cv   = float(sent_std / (sent_lens.mean() + 1e-9))

    # Paragraph length consistency
    raw_paras = [p.strip() for p in text.split('\n') if p.strip()]
    para_lens = np.array([len(p) for p in raw_paras], dtype=float)
    para_std  = float(para_lens.std())  if len(para_lens) > 1 else 0.0
    para_cv   = float(para_std / (para_lens.mean() + 1e-9))

    # Paragraph word-count balance
    para_wc  = np.array([len(p.split()) for p in raw_paras], dtype=float)
    para_bal = float(para_wc.std()) if len(para_wc) > 1 else 0.0

    # Writing pace — words per minute
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
    """
    Where in the session did revisions happen?
    Session split into thirds; count Remove/Cut events in each third.
    """
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

    norm_times = (remove_events - t_min) / t_dur   # 0 = session start, 1 = end
    total      = len(norm_times)

    return {
        'revision_ratio_early': float((norm_times < 1/3).sum())                         / total,
        'revision_ratio_mid':   float(((norm_times >= 1/3) & (norm_times < 2/3)).sum()) / total,
        'revision_ratio_late':  float((norm_times >= 2/3).sum())                        / total,
        'revision_timing_mean': float(norm_times.mean()),
        'revision_timing_std':  float(norm_times.std()) if total > 1 else 0.0,
    }


# =============================================================================
# COMPUTE FEATURES  (two-pass: per-essay → corpus TF-IDF)
# =============================================================================

def compute_features(
    logs: pd.DataFrame,
    tfidf_pipeline=None,
):
    """
    Compute per-essay features from raw keystroke logs.

    Two-pass design:
      Pass 1 — compute all per-essay scalar features and collect reconstructed texts.
      Pass 2 — fit (or apply) TF-IDF + TruncatedSVD on the corpus, then merge SVD
               components back into the feature table.

    Parameters
    ----------
    logs : pd.DataFrame
        Raw keystroke logs with columns: id, event_id, down_time, up_time,
        action_time, activity, down_event, up_event, text_change,
        cursor_position, word_count.
    tfidf_pipeline : fitted sklearn Pipeline or None
        If None, a new TF-IDF SVD pipeline is *fitted* on the passed corpus
        (training use).  If provided, the pipeline is used as-is for
        *transform* only (test/inference use).

    Returns
    -------
    tuple (pd.DataFrame, Pipeline)
        df            — one row per essay, columns = ['id'] + FEATURE_COLS
        tfidf_pipeline — the fitted Pipeline (same object if one was passed in)
    """
    logs = logs.sort_values(['id', 'event_id']).reset_index(drop=True)

    rows  = []
    texts = {}   # essay_id → reconstructed text (needed for TF-IDF pass)

    # ── Pass 1: per-essay scalar features ─────────────────────────────────────
    for essay_id, essay in logs.groupby('id'):
        essay = essay.reset_index(drop=True)
        row   = {'id': essay_id}

        # Steps 1–11 (same as v2)
        row.update(count_features(essay))
        row.update(input_word_features(essay))
        row.update(timing_features(essay))
        row.update(idle_features(essay))
        row.update(p_burst_features(essay))
        row.update(r_burst_features(essay))

        non_nonproduction = essay[essay['activity'] != 'Nonproduction']
        text = reconstruct_essay(non_nonproduction)
        texts[essay_id] = text

        row.update(word_features(text))
        row.update(sentence_features(text))
        row.update(paragraph_features(text))
        row.update(efficiency_features(essay, text))

        # Step 12 — session duration (same as v2)
        row['session_duration_sec'] = (
            row.get('down_time_max', 0) - row.get('down_time_min', 0)
        ) / 1000

        # Step 13 — per-word normalisation (same as v2, no verbosity_per_word)
        row.update(_normalize_counts(row))

        # Step 14 — readability + consistency + pace  [v4 new]
        row.update(_readability_features(text, row))

        # Step 15 — revision timing  [v4 new]
        row.update(_revision_timing_features(essay))

        rows.append(row)

    df = pd.DataFrame(rows).fillna(0)

    # ── Pass 2: TF-IDF SVD on reconstructed texts ─────────────────────────────
    # Character n-grams on anonymised text capture punctuation rhythm.
    corpus = [texts[eid] for eid in df['id']]

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
        svd_matrix = tfidf_pipeline.fit_transform(corpus)   # fit on training corpus
    else:
        svd_matrix = tfidf_pipeline.transform(corpus)       # transform only (test)

    svd_df       = pd.DataFrame(svd_matrix, columns=TFIDF_SVD_COLS)
    svd_df['id'] = df['id'].values

    df = df.merge(svd_df, on='id', how='left')

    return df[['id'] + FEATURE_COLS].fillna(0), tfidf_pipeline


# =============================================================================
# FEATURE COLUMNS  (v2 base 119 + 8 readability + 5 revision timing + 20 TF-IDF = 152)
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
    'input_word_length_skew',
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

    # ── 5. TYPING SPEED & FLUENCY ─────────────────────────────────────────────
    'keys_per_second',
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
    'n_unique_text_change',
    'n_unique_down_event',

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
    'r_burst_mean',
    'r_burst_std',
    'r_burst_median',
    'r_burst_max',
    'r_burst_first',

    # ── 8. PRODUCTION FLOW & NAVIGATION ──────────────────────────────────────
    'input_per_word',
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
    'cursor_position_mean',
    'cursor_position_std',
    'cursor_position_median',
    'cursor_position_q1',
    'cursor_position_q3',
    'leftclick_per_word',
    'arrowleft_per_word',
    'arrowright_per_word',
    'down_time_min',
    'down_time_mean',
    'down_time_std',
    'down_time_median',
    'down_time_q1',
    'down_time_q3',
    'down_time_max',
    'up_time_min',
    'up_time_max',

    # ── 9. READABILITY & STRUCTURAL CONSISTENCY  [v4 new] ─────────────────────
    'ari_score',              # Automated Readability Index
    'coleman_liau_score',     # Coleman-Liau readability index
    'sent_len_std',           # std of sentence lengths
    'sent_len_cv',            # coefficient of variation of sentence lengths
    'para_len_std',           # std of paragraph lengths
    'para_len_cv',            # coefficient of variation of paragraph lengths
    'para_balance',           # std of paragraph word counts
    'words_per_minute',       # gross writing speed

    # ── 10. REVISION TIMING  [v4 new] ─────────────────────────────────────────
    'revision_ratio_early',   # share of revisions in first third of session
    'revision_ratio_mid',     # share of revisions in middle third
    'revision_ratio_late',    # share of revisions in last third
    'revision_timing_mean',   # mean normalised revision timestamp
    'revision_timing_std',    # std of normalised revision timestamps

    # ── 11. TF-IDF SVD (char bigrams on reconstructed text)  [v4 new] ─────────
    *TFIDF_SVD_COLS,           # tfidf_svd_0 … tfidf_svd_19
]


# =============================================================================
# CATEGORIES (feature → group label, for importance plot colouring)
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
    'input_word_length_skew':     'Word Characteristics',
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
    'paragraph_len_mean':             'Paragraph Characteristics',
    'paragraph_len_median':           'Paragraph Characteristics',
    'paragraph_len_min':              'Paragraph Characteristics',
    'paragraph_len_max':              'Paragraph Characteristics',
    'paragraph_len_first':            'Paragraph Characteristics',
    'paragraph_len_last':             'Paragraph Characteristics',
    'paragraph_len_q1':               'Paragraph Characteristics',
    'paragraph_len_q3':               'Paragraph Characteristics',
    'paragraph_count':                'Paragraph Characteristics',
    'paragraph_word_count_mean':      'Paragraph Characteristics',
    'paragraph_word_count_min':       'Paragraph Characteristics',
    'paragraph_word_count_max':       'Paragraph Characteristics',
    'paragraph_word_count_first':     'Paragraph Characteristics',
    'paragraph_word_count_last':      'Paragraph Characteristics',
    'paragraph_word_count_q1':        'Paragraph Characteristics',
    'paragraph_word_count_median':    'Paragraph Characteristics',
    'paragraph_word_count_q3':        'Paragraph Characteristics',

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

    # 5. Typing Speed & Fluency
    'keys_per_second':            'Typing Speed & Fluency',
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
    'n_unique_text_change':       'Typing Speed & Fluency',
    'n_unique_down_event':        'Typing Speed & Fluency',

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
    'r_burst_mean':               'Revision Behavior',
    'r_burst_std':                'Revision Behavior',
    'r_burst_median':             'Revision Behavior',
    'r_burst_max':                'Revision Behavior',
    'r_burst_first':              'Revision Behavior',

    # 8. Production Flow & Navigation
    'input_per_word':             'Production Flow & Navigation',
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
    'cursor_position_mean':       'Production Flow & Navigation',
    'cursor_position_std':        'Production Flow & Navigation',
    'cursor_position_median':     'Production Flow & Navigation',
    'cursor_position_q1':         'Production Flow & Navigation',
    'cursor_position_q3':         'Production Flow & Navigation',
    'leftclick_per_word':         'Production Flow & Navigation',
    'arrowleft_per_word':         'Production Flow & Navigation',
    'arrowright_per_word':        'Production Flow & Navigation',
    'down_time_min':              'Production Flow & Navigation',
    'down_time_mean':             'Production Flow & Navigation',
    'down_time_std':              'Production Flow & Navigation',
    'down_time_median':           'Production Flow & Navigation',
    'down_time_q1':               'Production Flow & Navigation',
    'down_time_q3':               'Production Flow & Navigation',
    'down_time_max':              'Production Flow & Navigation',
    'up_time_min':                'Production Flow & Navigation',
    'up_time_max':                'Production Flow & Navigation',

    # 9. Readability & Structural Consistency  [v4 new]
    'ari_score':                  'Readability & Consistency',
    'coleman_liau_score':         'Readability & Consistency',
    'sent_len_std':               'Readability & Consistency',
    'sent_len_cv':                'Readability & Consistency',
    'para_len_std':               'Readability & Consistency',
    'para_len_cv':                'Readability & Consistency',
    'para_balance':               'Readability & Consistency',
    'words_per_minute':           'Readability & Consistency',

    # 10. Revision Timing  [v4 new]
    'revision_ratio_early':       'Revision Timing',
    'revision_ratio_mid':         'Revision Timing',
    'revision_ratio_late':        'Revision Timing',
    'revision_timing_mean':       'Revision Timing',
    'revision_timing_std':        'Revision Timing',

    # 11. TF-IDF SVD  [v4 new]
    **{col: 'TF-IDF SVD' for col in TFIDF_SVD_COLS},
}

# =============================================================================
# COLOUR PALETTE
# =============================================================================

CAT_PALETTE = {
    'Word Characteristics':              '#4C72B0',  # blue
    'Sentence Characteristics':          '#DD8452',  # orange
    'Paragraph Characteristics':         '#55A868',  # green
    'Punctuation & Formatting':          '#8172B2',  # purple
    'Typing Speed & Fluency':            '#C44E52',  # red
    'Pausing Behavior':                  '#937860',  # brown
    'Revision Behavior':                 '#DA8BC3',  # pink
    'Production Flow & Navigation':      '#8C8C8C',  # grey
    'Readability & Consistency':         '#E377C2',  # rose
    'Revision Timing':                   '#17BECF',  # cyan
    'TF-IDF SVD':                        '#BCBD22',  # yellow-green
}
