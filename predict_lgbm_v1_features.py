"""
Inference: Predict Essay Scores — Kaggle Submission (LightGBM, Important Features)
====================================================================================
Self-contained script: all feature engineering logic is inlined.
No local module imports required.

Feature set: important_features (raw counts — no per-word normalisation)

Paths (Kaggle notebook environment):
    DATA_DIR   = /kaggle/input/competitions/linking-writing-processes-to-writing-quality
    MODELS_DIR = /kaggle/input/datasets/sherlocked999/writing-quality-lightgbm-baseline
    OUTPUT_PATH = submission.csv
"""

import re
import glob
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import skew as _skew

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = '/kaggle/input/competitions/linking-writing-processes-to-writing-quality'
MODELS_DIR  = '/kaggle/input/datasets/sherlocked999/writing-quality-lightgbm-baseline'
OUTPUT_PATH = 'submission.csv'


# =============================================================================
# FEATURE COLUMNS  (important_features — raw counts, gain >= 15)
# =============================================================================

FEATURE_COLS = [

    # ── 1. WORD CHARACTERISTICS ───────────────────────────────────────────────
    'word_len_sum',               # total chars across all words — strongest essay-length proxy
    'word_len_mean',              # mean word length — indicator of vocabulary richness
    'word_len_q3',                # 75th pct word length — how long the longer words are
    'word_len_max',               # longest single word used
    'word_len_first',             # length of the first word typed
    'word_len_last',              # length of the last word typed
    'word_len_count',             # number of words — essentially the final word count
    'input_word_count',           # distinct words typed during the session
    'input_word_length_mean',     # mean typed word length (from keystroke sequence)
    'input_word_length_max',      # longest word typed during the session
    'input_word_length_std',      # std of typed word lengths — vocabulary variety
    'input_word_length_skew',     # skew of typed word lengths — long-tail word usage
    'word_count_max',             # max running word count = final essay length
    'word_count_std',             # std of running word count — pace of word accumulation
    'word_count_median',          # median running word count over the session
    'word_count_q1',              # 25th pct word count — how much was written early on
    'word_count_q3',              # 75th pct word count — word count near end of session

    # ── 2. SENTENCE CHARACTERISTICS ───────────────────────────────────────────
    'sent_len_sum',               # total chars across all sentences ≈ essay char length
    'sent_len_mean',              # mean sentence length — verbose vs. concise style
    'sent_len_median',            # median sentence length — robust to outlier sentences
    'sent_len_min',               # shortest sentence — detects fragments
    'sent_len_max',               # longest sentence — captures complex compound structures
    'sent_len_first',             # length of the opening sentence
    'sent_len_last',              # length of the closing sentence
    'sent_len_q1',                # 25th pct sentence length
    'sent_len_q3',                # 75th pct sentence length
    'sent_count',                 # number of sentences in the essay
    'sent_word_count_mean',       # mean words per sentence — syntactic density
    'sent_word_count_max',        # longest sentence by word count
    'sent_word_count_first',      # words in the first sentence
    'sent_word_count_last',       # words in the last sentence
    'sent_word_count_q1',         # 25th pct words per sentence
    'sent_word_count_median',     # median words per sentence
    'sent_word_count_q3',         # 75th pct words per sentence

    # ── 3. PARAGRAPH CHARACTERISTICS ──────────────────────────────────────────
    'paragraph_len_sum',          # total chars across all paragraphs ≈ essay length
    'paragraph_len_mean',         # mean paragraph length — how developed each idea is
    'paragraph_len_median',       # median paragraph length — robust central tendency
    'paragraph_len_min',          # shortest paragraph — thin or transitional paragraphs
    'paragraph_len_max',          # longest paragraph — most developed argument
    'paragraph_len_first',        # length of the introduction paragraph
    'paragraph_len_last',         # length of the conclusion paragraph
    'paragraph_len_q1',           # 25th pct paragraph length
    'paragraph_len_q3',           # 75th pct paragraph length
    'paragraph_count',            # total number of paragraphs — structural organisation
    'paragraph_word_count_mean',  # mean words per paragraph
    'paragraph_word_count_min',   # fewest words in any paragraph
    'paragraph_word_count_max',   # most words in any paragraph
    'paragraph_word_count_first', # word count of the introduction
    'paragraph_word_count_last',  # word count of the conclusion
    'paragraph_word_count_q1',    # 25th pct paragraph word count
    'paragraph_word_count_median',# median paragraph word count
    'paragraph_word_count_q3',    # 75th pct paragraph word count
    'paragraph_word_count_sum',   # total words across all paragraphs ≈ word_count_max

    # ── 4. PUNCTUATION & FORMATTING ───────────────────────────────────────────
    'down_event_comma_cnt',       # comma key presses
    'text_change_comma_cnt',      # literal ',' inserted
    'up_event_comma_cnt',         # comma key releases
    'down_event_period_cnt',      # period key presses
    'text_change_period_cnt',     # literal '.' inserted
    'down_event_Enter_cnt',       # enter presses
    'text_change_newline_cnt',    # '\n' inserted
    'down_event_Shift_cnt',       # shift presses
    'down_event_CapsLock_cnt',    # caps-lock usage
    'text_change_dash_cnt',       # '-' inserted
    'text_change_question_cnt',   # '?' inserted
    'text_change_quote_cnt',      # "'" inserted

    # ── 5. TYPING SPEED & FLUENCY ─────────────────────────────────────────────
    'keys_per_second',            # gross keystrokes per second
    'action_time_sum',            # total key-hold time
    'action_time_mean',           # mean key-hold duration
    'action_time_std',            # std of hold durations
    'action_time_median',         # median hold duration
    'action_time_max',            # longest single key hold
    'action_time_q1',             # 25th pct hold duration
    'action_time_q3',             # 75th pct hold duration
    'down_event_q_cnt',           # total letter keystrokes
    'text_change_q_cnt',          # anonymised letter insertions
    'up_event_q_cnt',             # letter key releases
    'down_event_Space_cnt',       # space presses
    'text_change_space_cnt',      # spaces inserted
    'n_unique_text_change',       # distinct text changes
    'n_unique_down_event',        # distinct keys used
    'n_unique_up_event',          # distinct key releases used

    # ── 6. PAUSING BEHAVIOR ───────────────────────────────────────────────────
    'idle_largest_latency',       # single longest idle gap
    'idle_median_latency',        # median idle gap
    'idle_mean',                  # mean idle gap
    'idle_std',                   # std of idle gaps
    'idle_total',                 # total time spent idle
    'pauses_half_sec',            # pauses > 0.5 s
    'pauses_1_sec',               # pauses > 1 s
    'pauses_1_half_sec',          # pauses > 1.5 s
    'pauses_2_sec',               # pauses > 2 s
    'pauses_3_sec',               # pauses > 3 s

    # ── 7. REVISION BEHAVIOR ──────────────────────────────────────────────────
    'activity_RemoveCut_cnt',     # total Remove/Cut events
    'down_event_Backspace_cnt',   # backspace presses
    'r_burst_mean',               # mean revision burst length
    'r_burst_std',                # std of revision burst lengths
    'r_burst_median',             # median revision burst length
    'r_burst_max',                # largest single deletion burst
    'r_burst_first',              # first revision burst

    # ── 8. PRODUCTION FLOW & NAVIGATION ──────────────────────────────────────
    'activity_Input_cnt',         # total input events
    'activity_Nonproduction_cnt', # navigation/thinking events
    'p_burst_count',              # number of production bursts
    'p_burst_mean',               # mean burst length
    'p_burst_std',                # std of burst lengths
    'p_burst_median',             # median burst length
    'p_burst_max',                # longest single burst
    'p_burst_first',              # first burst length
    'p_burst_last',               # last burst length
    'product_to_keys',            # net chars produced ÷ total keys pressed
    'cursor_position_max',        # furthest cursor position ≈ essay length
    'cursor_position_mean',       # mean cursor position
    'cursor_position_std',        # std of cursor position
    'cursor_position_median',     # median cursor position
    'cursor_position_q1',         # 25th pct cursor position
    'cursor_position_q3',         # 75th pct cursor position
    'down_event_Leftclick_cnt',   # mouse clicks
    'down_event_ArrowLeft_cnt',   # left-arrow presses
    'down_event_ArrowRight_cnt',  # right-arrow presses
    'down_time_min',              # timestamp of first keypress
    'down_time_mean',             # mean keypress timestamp
    'down_time_std',              # std of keypress timestamps
    'down_time_median',           # median keypress timestamp
    'down_time_q1',               # 25th pct keypress time
    'down_time_q3',               # 75th pct keypress time
    'down_time_max',              # timestamp of last keypress
    'up_time_min',                # first key-release time
    'up_time_max',                # last key-release time
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
    ]:
        feats[key] = int((act == value).sum())

    for value, key in [
        ('q',  'text_change_q_cnt'),
        (' ',  'text_change_space_cnt'),
        ('.',  'text_change_period_cnt'),
        (',',  'text_change_comma_cnt'),
        ('\n', 'text_change_newline_cnt'),
        ("'",  'text_change_quote_cnt'),
        ('-',  'text_change_dash_cnt'),
        ('?',  'text_change_question_cnt'),
    ]:
        feats[key] = int((tc == value).sum())

    for value, key in [
        ('q',          'down_event_q_cnt'),
        ('Space',      'down_event_Space_cnt'),
        ('Backspace',  'down_event_Backspace_cnt'),
        ('Shift',      'down_event_Shift_cnt'),
        ('ArrowRight', 'down_event_ArrowRight_cnt'),
        ('Leftclick',  'down_event_Leftclick_cnt'),
        ('ArrowLeft',  'down_event_ArrowLeft_cnt'),
        ('.',          'down_event_period_cnt'),
        (',',          'down_event_comma_cnt'),
        ('Enter',      'down_event_Enter_cnt'),
        ('CapsLock',   'down_event_CapsLock_cnt'),
    ]:
        feats[key] = int((de == value).sum())

    for value, key in [
        ('q', 'up_event_q_cnt'),
        (',', 'up_event_comma_cnt'),
    ]:
        feats[key] = int((ue == value).sum())

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
        'input_word_count':        len(lengths),
        'input_word_length_mean':  np.mean(lengths)      if lengths else 0.0,
        'input_word_length_max':   np.max(lengths)       if lengths else 0.0,
        'input_word_length_std':   np.std(lengths)       if lengths else 0.0,
        'input_word_length_skew':  float(_skew(lengths)) if lengths else 0.0,
    }


# ── Step 3: Timing features ───────────────────────────────────────────────────

def _timing_features(essay: pd.DataFrame) -> dict:
    feats = {}

    at = essay['action_time']
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
    feats['up_time_min'] = float(ut.min())
    feats['up_time_max'] = float(ut.max())

    cp = essay['cursor_position']
    feats['cursor_position_mean']   = float(cp.mean())
    feats['cursor_position_std']    = float(cp.std())
    feats['cursor_position_max']    = float(cp.max())
    feats['cursor_position_median'] = float(cp.median())
    feats['cursor_position_q1']     = float(cp.quantile(0.25))
    feats['cursor_position_q3']     = float(cp.quantile(0.75))

    wc = essay['word_count']
    feats['word_count_max']    = float(wc.max())
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
        'idle_largest_latency': float(gaps.max())    if len(gaps) else 0.0,
        'idle_median_latency':  float(gaps.median()) if len(gaps) else 0.0,
        'idle_mean':            float(gaps.mean())   if len(gaps) else 0.0,
        'idle_std':             float(gaps.std())    if len(gaps) else 0.0,
        'idle_total':           float(gaps.sum())    if len(gaps) else 0.0,
        'pauses_half_sec':      int(((gaps > 0.5) & (gaps < 1.0)).sum()),
        'pauses_1_sec':         int(((gaps > 1.0) & (gaps < 1.5)).sum()),
        'pauses_1_half_sec':    int(((gaps > 1.5) & (gaps < 2.0)).sum()),
        'pauses_2_sec':         int(((gaps > 2.0) & (gaps < 3.0)).sum()),
        'pauses_3_sec':         int((gaps > 3.0).sum()),
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
        names = ['count', 'sum', 'mean', 'min', 'max', 'first', 'last', 'q1', 'median', 'q3']
        return {f'{prefix}_{n}': 0.0 for n in names}
    arr = np.array(values, dtype=float)
    return {
        f'{prefix}_count':  len(arr),
        f'{prefix}_sum':    arr.sum(),
        f'{prefix}_mean':   arr.mean(),
        f'{prefix}_min':    arr.min(),
        f'{prefix}_max':    arr.max(),
        f'{prefix}_first':  arr[0],
        f'{prefix}_last':   arr[-1],
        f'{prefix}_q1':     np.percentile(arr, 25),
        f'{prefix}_median': np.median(arr),
        f'{prefix}_q3':     np.percentile(arr, 75),
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


# ── Main pipeline ─────────────────────────────────────────────────────────────

def compute_features(logs: pd.DataFrame) -> pd.DataFrame:
    logs = logs.sort_values(['id', 'event_id']).reset_index(drop=True)

    rows = []
    for essay_id, essay in logs.groupby('id'):
        essay = essay.reset_index(drop=True)
        row   = {'id': essay_id}

        # Step 1 — count specific keys, activities, and text-change characters
        row.update(_count_features(essay))

        # Step 2 — word-length stats from typed q-sequences
        row.update(_input_word_features(essay))

        # Step 3 — aggregate action_time, down_time, up_time, cursor, word_count
        row.update(_timing_features(essay))

        # Step 4 — idle gaps between keys and pause counts at various thresholds
        row.update(_idle_features(essay))

        # Step 5 — production burst lengths (runs of typing < 2 s apart)
        row.update(_p_burst_features(essay))

        # Step 6 — revision burst lengths (consecutive Remove/Cut runs)
        row.update(_r_burst_features(essay))

        # Step 7 — replay all events to recover the final essay text
        non_nonproduction = essay[essay['activity'] != 'Nonproduction']
        text = _reconstruct_essay(non_nonproduction)

        # Step 8 — word length stats from the reconstructed essay text
        row.update(_word_features(text))

        # Step 9 — sentence length and word-count stats from reconstructed text
        row.update(_sentence_features(text))

        # Step 10 — paragraph length and word-count stats from reconstructed text
        row.update(_paragraph_features(text))

        # Step 11 — efficiency ratios
        row.update(_efficiency_features(essay, text))

        rows.append(row)

    return pd.DataFrame(rows).fillna(0)


# =============================================================================
# SECTION 1 – LOAD MODELS
# =============================================================================
model_paths = sorted(glob.glob(f'{MODELS_DIR}/lgbm_s*_fold*.txt'))
if not model_paths:
    raise FileNotFoundError(
        f"No fold models found in {MODELS_DIR}/\n"
        "Upload lgbm_s42_fold1.txt … lgbm_s46_fold10.txt to the dataset."
    )

models = [lgb.Booster(model_file=p) for p in model_paths]
print(f"Loaded {len(models)} models from {MODELS_DIR}/ (average of all for final prediction)")

# =============================================================================
# SECTION 2 – FEATURE ENGINEERING
# =============================================================================
print("Loading test_logs.csv...")
logs = pd.read_csv(f'{DATA_DIR}/test_logs.csv')
print(f"  {logs.shape[0]:,} rows | {logs['id'].nunique()} essays")

print("Computing per-essay features...")
test_df = compute_features(logs).fillna(0)
X_test  = test_df[FEATURE_COLS].values
print(f"  Feature matrix: {X_test.shape[0]} essays × {X_test.shape[1]} features")

# =============================================================================
# SECTION 3 – INFERENCE
# =============================================================================
print("\nRunning inference...")
fold_preds = np.stack([model.predict(X_test) for model in models], axis=0)
preds      = fold_preds.mean(axis=0)
preds      = np.clip(preds, 0.5, 6.0)

print(f"  Prediction stats — mean: {preds.mean():.3f} | "
      f"std: {preds.std():.3f} | min: {preds.min():.3f} | max: {preds.max():.3f}")

# =============================================================================
# SECTION 4 – WRITE SUBMISSION
# =============================================================================
submission = pd.DataFrame({'id': test_df['id'], 'score': preds})
submission.to_csv(OUTPUT_PATH, index=False)
print(f"\nSubmission saved to: {OUTPUT_PATH}")
print(f"  {len(submission)} rows")
print(submission.head(10).to_string(index=False))
