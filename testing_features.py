# For testing new features or removing old features
"""
Important Features: Linking Writing Process to Writing Quality
==============================================================
Pipeline stages (sequential)
-----------------------------
  Step 1  count_features       — count specific keys, activities, text changes
  Step 2  input_word_features  — word-length stats from typed q-sequences
  Step 3  timing_features      — action/down/up time, cursor, word-count aggs
  Step 4  idle_features        — inter-key idle gaps and pause-count thresholds
  Step 5  p_burst_features     — production burst lengths (writing flow episodes)
  Step 6  r_burst_features     — revision burst lengths (deletion episodes)
  Step 7  reconstruct_essay    — replay keystrokes to recover final essay text
  Step 8  word_features        — word length stats from reconstructed text
  Step 9  sentence_features    — sentence length and word-count stats
  Step 10 paragraph_features   — paragraph length and word-count stats
  Step 11 efficiency_features  — product-to-keys ratio and keys-per-second


Categories:
    1. Word Characteristics      — length distributions of individual words
    2. Sentence Characteristics  — length and word-count stats per sentence
    3. Paragraph Characteristics — length and word-count stats per paragraph
    4. Punctuation & Formatting  — punctuation marks, capitalisation, newlines
    5. Typing Speed & Fluency    — keystroke speed, rhythm, and key diversity
    6. Pausing Behavior          — idle gaps and pause-count thresholds
    7. Revision Behavior         — deletions, backspace, revision bursts
    8. Production Flow           — writing bursts, efficiency, cursor/navigation

Usage:
    from important_features import compute_features, FEATURE_COLS, CATEGORIES, CAT_PALETTE
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))  # ensure writing_process/ is on the path

import pandas as pd

from preprocess.essay_reconstruction import reconstruct_essay
from preprocess.keystroke_stats      import count_features, input_word_features, timing_features
from preprocess.burst_stats          import idle_features, p_burst_features, r_burst_features
from preprocess.text_stats           import word_features, sentence_features, paragraph_features
from preprocess.efficiency           import efficiency_features


# =============================================================================
# COMPUTE FEATURES
# =============================================================================

def _normalize_counts(row: dict) -> dict:
    """
    Normalize count-based features per word to reduce essay-length bias.
    Uses max(word_len_count, 1) to avoid division by zero.
    """
    word_count = max(row.get('word_len_count', 1), 1)
    
    normalized = {}
    
    # Punctuation & Formatting (per-word)
    normalized['comma_per_word']      = row.get('down_event_comma_cnt', 0) / word_count
    normalized['comma_text_per_word']  = row.get('text_change_comma_cnt', 0) / word_count
    normalized['period_per_word']      = row.get('down_event_period_cnt', 0) / word_count
    normalized['period_text_per_word'] = row.get('text_change_period_cnt', 0) / word_count
    normalized['enter_per_word']      = row.get('down_event_Enter_cnt', 0) / word_count
    normalized['newline_per_word']     = row.get('text_change_newline_cnt', 0) / word_count
    normalized['shift_per_word']       = row.get('down_event_Shift_cnt', 0) / word_count
    normalized['capslock_per_word']    = row.get('down_event_CapsLock_cnt', 0) / word_count
    normalized['dash_per_word']        = row.get('text_change_dash_cnt', 0) / word_count
    normalized['question_per_word']    = row.get('text_change_question_cnt', 0) / word_count
    normalized['quote_per_word']       = row.get('text_change_quote_cnt', 0) / word_count
    
    # Typing Speed (per-word)
    normalized['q_keys_per_word']      = row.get('down_event_q_cnt', 0) / word_count
    normalized['q_text_per_word']      = row.get('text_change_q_cnt', 0) / word_count
    normalized['space_per_word']       = row.get('down_event_Space_cnt', 0) / word_count
    normalized['space_text_per_word']  = row.get('text_change_space_cnt', 0) / word_count
    
    # Pausing Behavior (per-word)
    normalized['pauses_half_per_word']  = row.get('pauses_half_sec', 0) / word_count
    normalized['pauses_1_per_word']     = row.get('pauses_1_sec', 0) / word_count
    normalized['pauses_1half_per_word']= row.get('pauses_1_half_sec', 0) / word_count
    normalized['pauses_2_per_word']    = row.get('pauses_2_sec', 0) / word_count
    normalized['pauses_3_per_word']    = row.get('pauses_3_sec', 0) / word_count
    
    # Revision Behavior (per-word)
    normalized['removecut_per_word']   = row.get('activity_RemoveCut_cnt', 0) / word_count
    normalized['backspace_per_word']   = row.get('down_event_Backspace_cnt', 0) / word_count
    
    # Production Flow (per-word)
    normalized['input_per_word']       = row.get('activity_Input_cnt', 0) / word_count
    normalized['nonproduction_per_word']= row.get('activity_Nonproduction_cnt', 0) / word_count
    normalized['leftclick_per_word']   = row.get('down_event_Leftclick_cnt', 0) / word_count
    normalized['arrowleft_per_word']   = row.get('down_event_ArrowLeft_cnt', 0) / word_count
    normalized['arrowright_per_word']  = row.get('down_event_ArrowRight_cnt', 0) / word_count
    
    # Sentence Count (per-word)
    normalized['sent_per_word']         = row.get('sent_count', 0) / word_count
    
    # Burst Count (per-word)
    normalized['p_burst_per_word']      = row.get('p_burst_count', 0) / word_count
    
    return normalized


def compute_features(logs: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-essay features from raw keystroke logs.

    Parameters
    ----------
    logs : pd.DataFrame
        Raw logs with columns: id, event_id, down_time, up_time, action_time,
        activity, down_event, up_event, text_change, cursor_position, word_count.

    Returns
    -------
    pd.DataFrame
        One row per essay; columns = ['id'] + FEATURE_COLS.
    """
    logs = logs.sort_values(['id', 'event_id']).reset_index(drop=True)

    rows = []
    for essay_id, essay in logs.groupby('id'):
        essay = essay.reset_index(drop=True)
        row   = {'id': essay_id}

        # Step 1 — count specific keys, activities, and text-change characters
        row.update(count_features(essay))

        # Step 2 — word-length stats from typed q-sequences (no reconstruction needed)
        row.update(input_word_features(essay))

        # Step 3 — aggregate action_time, down_time, up_time, cursor, word_count
        row.update(timing_features(essay))

        # Step 4 — idle gaps between keys and pause counts at various thresholds
        row.update(idle_features(essay))

        # Step 5 — production burst lengths (runs of typing < 2 s apart)
        row.update(p_burst_features(essay))

        # Step 6 — revision burst lengths (consecutive Remove/Cut runs)
        row.update(r_burst_features(essay))

        # Step 7 — replay all events to recover the final essay text
        non_nonproduction = essay[essay['activity'] != 'Nonproduction']
        text = reconstruct_essay(non_nonproduction)

        # Step 8 — word length stats from the reconstructed essay text
        row.update(word_features(text))

        # Step 9 — sentence length and word-count stats from reconstructed text
        row.update(sentence_features(text))

        # Step 10 — paragraph length and word-count stats from reconstructed text
        row.update(paragraph_features(text))

        # Step 11 — efficiency ratios (need both essay df and reconstructed text)
        row.update(efficiency_features(essay, text))

        # Step 12 — session duration
        row['session_duration_sec'] = (row.get('down_time_max', 0) - row.get('down_time_min', 0)) / 1000

        # Step 13 — normalize count-based features per word
        row.update(_normalize_counts(row))

        rows.append(row)

    result = pd.DataFrame(rows).fillna(0)
    return result[['id'] + FEATURE_COLS]

# =============================================================================
# FEATURE COLUMNS  (grouped by concept, gain >= 15)
# =============================================================================

FEATURE_COLS = [

    # ── 1. WORD CHARACTERISTICS ───────────────────────────────────────────────
    # word_len_* : from reconstructed final text. input_word_length_* : from
    # typed q-sequences during the session. word_count_* : running word total.
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
    'word_count_std',             # std of running word count — pace of word accumulation
    'word_count_median',          # median running word count over the session
    'word_count_q1',              # 25th pct word count — how much was written early on
    'word_count_q3',              # 75th pct word count — word count near end of session

    # ── 2. SENTENCE CHARACTERISTICS ───────────────────────────────────────────
    # sent_len_* : sentence length in chars. sent_word_count_* : words per sentence.
    'sent_len_mean',              # mean sentence length — verbose vs. concise style
    'sent_len_median',            # median sentence length — robust to outlier sentences
    'sent_len_min',               # shortest sentence — detects fragments
    'sent_len_max',               # longest sentence — captures complex compound structures
    'sent_len_first',             # length of the opening sentence
    'sent_len_last',              # length of the closing sentence
    'sent_len_q1',                # 25th pct sentence length
    'sent_len_q3',                # 75th pct sentence length
    'sent_per_word',              # number of sentences per word
    'sent_word_count_mean',       # mean words per sentence — syntactic density
    'sent_word_count_max',        # longest sentence by word count
    'sent_word_count_first',      # words in the first sentence
    'sent_word_count_last',       # words in the last sentence
    'sent_word_count_q1',         # 25th pct words per sentence
    'sent_word_count_median',     # median words per sentence
    'sent_word_count_q3',         # 75th pct words per sentence

    # ── 3. PARAGRAPH CHARACTERISTICS ──────────────────────────────────────────
    # paragraph_len_* : paragraph length in chars. paragraph_word_count_* : words per paragraph.
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
    # ── 4. PUNCTUATION & FORMATTING ───────────────────────────────────────────
    # Letters are anonymised as 'q' but punctuation in text_change is preserved.
    # Raw counts normalized per-word to reduce essay-length bias.
    'comma_per_word',             # comma key presses per word
    'comma_text_per_word',        # literal ',' inserted per word
    'period_per_word',            # period key presses per word
    'period_text_per_word',       # literal '.' inserted per word
    'enter_per_word',             # enter presses per word
    'newline_per_word',            # '\n' inserted per word
    'shift_per_word',             # shift presses per word
    'capslock_per_word',          # caps-lock usage per word
    'dash_per_word',              # '-' inserted per word
    'question_per_word',          # '?' inserted per word
    'quote_per_word',             # "'" inserted per word

    # ── 5. TYPING SPEED & FLUENCY ─────────────────────────────────────────────
    # action_time_* : duration the key was physically held (down → up).
    'keys_per_second',            # gross keystrokes per second — overall typing speed
    'action_time_sum',            # total time keys were physically held down
    'action_time_mean',           # mean key-hold duration — typical keystroke rhythm
    'action_time_std',            # std of hold durations — consistency of key presses
    'action_time_median',         # median hold duration — robust rhythm measure
    'action_time_max',            # longest single key hold (e.g. held backspace to delete)
    'action_time_q1',             # 25th pct hold duration
    'action_time_q3',             # 75th pct hold duration
    'q_keys_per_word',            # letter keystrokes per word
    'q_text_per_word',            # anonymised letter insertions per word
    'space_per_word',             # space presses per word
    'space_text_per_word',        # spaces inserted per word
    'n_unique_text_change',       # distinct text changes — typing content variety
    'n_unique_down_event',        # distinct keys used — keyboard range

    # ── 6. PAUSING BEHAVIOR ───────────────────────────────────────────────────
    # Inter-key idle gap = time from key release to the next key press.
    'idle_largest_latency',       # single longest idle gap — biggest pause taken
    'idle_median_latency',        # median idle gap — typical thinking-break length
    'idle_mean',                  # mean idle gap — overall session pacing
    'idle_std',                   # std of idle gaps — consistency of typing rhythm
    'idle_total',                 # total time spent idle — time not actively typing
    'pauses_half_per_word',       # pauses > 0.5 s per word
    'pauses_1_per_word',          # pauses > 1 s per word
    'pauses_1half_per_word',      # pauses > 1.5 s per word
    'pauses_2_per_word',          # pauses > 2 s per word
    'pauses_3_per_word',          # pauses > 3 s per word

    # ── 7. REVISION BEHAVIOR ──────────────────────────────────────────────────
    # R-burst = consecutive Remove/Cut run — a focused deletion episode.
    'removecut_per_word',         # Remove/Cut events per word
    'backspace_per_word',          # backspace presses per word
    'r_burst_mean',               # mean revision burst length — typical deletion run size
    'r_burst_std',                # std of revision burst lengths — consistency of deletions
    'r_burst_median',             # median revision burst length
    'r_burst_max',                # largest single deletion burst — big structural revision
    'r_burst_first',              # first revision burst — how soon writer started correcting

    # ── 8. PRODUCTION FLOW & NAVIGATION ──────────────────────────────────────
    # P-burst = run of Input + Remove/Cut events < 2 s apart — a writing flow episode.
    'input_per_word',             # input events per word
    'nonproduction_per_word',    # navigation/thinking events per word
    'p_burst_per_word',           # production bursts per word
    'p_burst_mean',               # mean burst length — typical flow episode size
    'p_burst_std',                # std of burst lengths — consistency of flow
    'p_burst_median',             # median burst length
    'p_burst_max',                # longest single burst — peak flow episode
    'p_burst_first',              # first burst length — how fast writer entered flow
    'p_burst_last',               # last burst length — late-session production effort
    'product_to_keys',            # net chars produced ÷ total keys pressed — efficiency
    'session_duration_sec',        # total session duration in seconds
    'cursor_position_mean',       # mean cursor position — average depth into essay
    'cursor_position_std',        # std of cursor position — how much cursor moved around
    'cursor_position_median',     # median cursor position
    'cursor_position_q1',         # 25th pct cursor position
    'cursor_position_q3',         # 75th pct cursor position
    'leftclick_per_word',         # mouse clicks per word
    'arrowleft_per_word',         # left-arrow presses per word
    'arrowright_per_word',        # right-arrow presses per word
    'down_time_min',              # timestamp of first keypress — session start offset
    'down_time_mean',             # mean keypress timestamp — session pacing centroid
    'down_time_std',              # std of keypress timestamps — spread of activity
    'down_time_median',           # median keypress timestamp
    'down_time_q1',               # 25th pct keypress time
    'down_time_q3',               # 75th pct keypress time
    'down_time_max',              # timestamp of last keypress — session end
    'up_time_min',                # first key-release time
    'up_time_max',                # last key-release time
]

# =============================================================================
# CATEGORY MAPPING  (one entry per feature in FEATURE_COLS)
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
    'nonproduction_per_word':    'Production Flow & Navigation',
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
    'leftclick_per_word':        'Production Flow & Navigation',
    'arrowleft_per_word':        'Production Flow & Navigation',
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
}

# =============================================================================
# COLOUR PALETTE  (one colour per conceptual category)
# =============================================================================

CAT_PALETTE = {
    'Word Characteristics':       '#4C72B0',  # blue
    'Sentence Characteristics':   '#DD8452',  # orange
    'Paragraph Characteristics':  '#55A868',  # green
    'Punctuation & Formatting':   '#8172B2',  # purple
    'Typing Speed & Fluency':     '#C44E52',  # red
    'Pausing Behavior':           '#937860',  # brown
    'Revision Behavior':          '#DA8BC3',  # pink
    'Production Flow & Navigation':'#8C8C8C', # grey
}
