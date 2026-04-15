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
    'word_count_max',             # max running word count = final essay length
    'word_count_std',             # std of running word count — pace of word accumulation
    'word_count_median',          # median running word count over the session
    'word_count_q1',              # 25th pct word count — how much was written early on
    'word_count_q3',              # 75th pct word count — word count near end of session

    # ── 2. SENTENCE CHARACTERISTICS ───────────────────────────────────────────
    # sent_len_* : sentence length in chars. sent_word_count_* : words per sentence.
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
    # paragraph_len_* : paragraph length in chars. paragraph_word_count_* : words per paragraph.
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
    # Letters are anonymised as 'q' but punctuation in text_change is preserved.
    'down_event_comma_cnt',       # comma key presses — syntactic complexity (clauses, lists)
    'text_change_comma_cnt',      # literal ',' inserted — confirms comma usage
    'up_event_comma_cnt',         # comma key releases — secondary comma signal
    'down_event_period_cnt',      # period key presses — sentence count proxy
    'text_change_period_cnt',     # literal '.' inserted — sentence ending count
    'down_event_Enter_cnt',       # enter presses — paragraph/line break creation
    'text_change_newline_cnt',    # '\n' inserted — paragraph boundary count
    'down_event_Shift_cnt',       # shift presses — capitalisation and punctuation (?,!,:)
    'down_event_CapsLock_cnt',    # caps-lock usage — stylistic or error signal
    'text_change_dash_cnt',       # '-' inserted — compound words, em-dashes, hyphens
    'text_change_question_cnt',   # '?' inserted — interrogative sentence usage
    'text_change_quote_cnt',      # "'" inserted — apostrophes and quotation marks

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
    'down_event_q_cnt',           # total letter keystrokes — raw character input volume
    'text_change_q_cnt',          # anonymised letter insertions — net typing volume
    'up_event_q_cnt',             # letter key releases — mirrors down_event_q_cnt
    'down_event_Space_cnt',       # space presses — word boundary count
    'text_change_space_cnt',      # spaces inserted — net word count signal
    'n_unique_text_change',       # distinct text changes — typing content variety
    'n_unique_down_event',        # distinct keys used — keyboard range
    'n_unique_up_event',          # distinct key releases used

    # ── 6. PAUSING BEHAVIOR ───────────────────────────────────────────────────
    # Inter-key idle gap = time from key release to the next key press.
    'idle_largest_latency',       # single longest idle gap — biggest pause taken
    'idle_median_latency',        # median idle gap — typical thinking-break length
    'idle_mean',                  # mean idle gap — overall session pacing
    'idle_std',                   # std of idle gaps — consistency of typing rhythm
    'idle_total',                 # total time spent idle — time not actively typing
    'pauses_half_sec',            # pauses > 0.5 s — micro hesitations between words
    'pauses_1_sec',               # pauses > 1 s — brief thinking breaks
    'pauses_1_half_sec',          # pauses > 1.5 s
    'pauses_2_sec',               # pauses > 2 s — deliberate planning or reading breaks
    'pauses_3_sec',               # pauses > 3 s — longer cognitive processing episodes

    # ── 7. REVISION BEHAVIOR ──────────────────────────────────────────────────
    # R-burst = consecutive Remove/Cut run — a focused deletion episode.
    'activity_RemoveCut_cnt',     # total Remove/Cut events — absolute revision volume
    'down_event_Backspace_cnt',   # backspace presses — local character-level corrections
    'r_burst_mean',               # mean revision burst length — typical deletion run size
    'r_burst_std',                # std of revision burst lengths — consistency of deletions
    'r_burst_median',             # median revision burst length
    'r_burst_max',                # largest single deletion burst — big structural revision
    'r_burst_first',              # first revision burst — how soon writer started correcting

    # ── 8. PRODUCTION FLOW & NAVIGATION ──────────────────────────────────────
    # P-burst = run of Input + Remove/Cut events < 2 s apart — a writing flow episode.
    'activity_Input_cnt',         # total input events — overall typing engagement
    'activity_Nonproduction_cnt', # navigation/thinking events not producing text
    'p_burst_count',              # number of production bursts — how many flow episodes
    'p_burst_mean',               # mean burst length — typical flow episode size
    'p_burst_std',                # std of burst lengths — consistency of flow
    'p_burst_median',             # median burst length
    'p_burst_max',                # longest single burst — peak flow episode
    'p_burst_first',              # first burst length — how fast writer entered flow
    'p_burst_last',               # last burst length — late-session production effort
    'product_to_keys',            # net chars produced ÷ total keys pressed — efficiency
    'cursor_position_max',        # furthest cursor position ≈ essay character length
    'cursor_position_mean',       # mean cursor position — average depth into essay
    'cursor_position_std',        # std of cursor position — how much cursor moved around
    'cursor_position_median',     # median cursor position
    'cursor_position_q1',         # 25th pct cursor position
    'cursor_position_q3',         # 75th pct cursor position
    'down_event_Leftclick_cnt',   # mouse clicks — jumping to a new position in text
    'down_event_ArrowLeft_cnt',   # left-arrow presses — backward character navigation
    'down_event_ArrowRight_cnt',  # right-arrow presses — forward character navigation
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
    'word_count_max':             'Word Characteristics',
    'word_count_std':             'Word Characteristics',
    'word_count_median':          'Word Characteristics',
    'word_count_q1':              'Word Characteristics',
    'word_count_q3':              'Word Characteristics',

    # 2. Sentence Characteristics
    'sent_len_sum':               'Sentence Characteristics',
    'sent_len_mean':              'Sentence Characteristics',
    'sent_len_median':            'Sentence Characteristics',
    'sent_len_min':               'Sentence Characteristics',
    'sent_len_max':               'Sentence Characteristics',
    'sent_len_first':             'Sentence Characteristics',
    'sent_len_last':              'Sentence Characteristics',
    'sent_len_q1':                'Sentence Characteristics',
    'sent_len_q3':                'Sentence Characteristics',
    'sent_count':                 'Sentence Characteristics',
    'sent_word_count_mean':       'Sentence Characteristics',
    'sent_word_count_max':        'Sentence Characteristics',
    'sent_word_count_first':      'Sentence Characteristics',
    'sent_word_count_last':       'Sentence Characteristics',
    'sent_word_count_q1':         'Sentence Characteristics',
    'sent_word_count_median':     'Sentence Characteristics',
    'sent_word_count_q3':         'Sentence Characteristics',

    # 3. Paragraph Characteristics
    'paragraph_len_sum':              'Paragraph Characteristics',
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
    'paragraph_word_count_sum':       'Paragraph Characteristics',

    # 4. Punctuation & Formatting
    'down_event_comma_cnt':       'Punctuation & Formatting',
    'text_change_comma_cnt':      'Punctuation & Formatting',
    'up_event_comma_cnt':         'Punctuation & Formatting',
    'down_event_period_cnt':      'Punctuation & Formatting',
    'text_change_period_cnt':     'Punctuation & Formatting',
    'down_event_Enter_cnt':       'Punctuation & Formatting',
    'text_change_newline_cnt':    'Punctuation & Formatting',
    'down_event_Shift_cnt':       'Punctuation & Formatting',
    'down_event_CapsLock_cnt':    'Punctuation & Formatting',
    'text_change_dash_cnt':       'Punctuation & Formatting',
    'text_change_question_cnt':   'Punctuation & Formatting',
    'text_change_quote_cnt':      'Punctuation & Formatting',

    # 5. Typing Speed & Fluency
    'keys_per_second':            'Typing Speed & Fluency',
    'action_time_sum':            'Typing Speed & Fluency',
    'action_time_mean':           'Typing Speed & Fluency',
    'action_time_std':            'Typing Speed & Fluency',
    'action_time_median':         'Typing Speed & Fluency',
    'action_time_max':            'Typing Speed & Fluency',
    'action_time_q1':             'Typing Speed & Fluency',
    'action_time_q3':             'Typing Speed & Fluency',
    'down_event_q_cnt':           'Typing Speed & Fluency',
    'text_change_q_cnt':          'Typing Speed & Fluency',
    'up_event_q_cnt':             'Typing Speed & Fluency',
    'down_event_Space_cnt':       'Typing Speed & Fluency',
    'text_change_space_cnt':      'Typing Speed & Fluency',
    'n_unique_text_change':       'Typing Speed & Fluency',
    'n_unique_down_event':        'Typing Speed & Fluency',
    'n_unique_up_event':          'Typing Speed & Fluency',

    # 6. Pausing Behavior
    'idle_largest_latency':       'Pausing Behavior',
    'idle_median_latency':        'Pausing Behavior',
    'idle_mean':                  'Pausing Behavior',
    'idle_std':                   'Pausing Behavior',
    'idle_total':                 'Pausing Behavior',
    'pauses_half_sec':            'Pausing Behavior',
    'pauses_1_sec':               'Pausing Behavior',
    'pauses_1_half_sec':          'Pausing Behavior',
    'pauses_2_sec':               'Pausing Behavior',
    'pauses_3_sec':               'Pausing Behavior',

    # 7. Revision Behavior
    'activity_RemoveCut_cnt':     'Revision Behavior',
    'down_event_Backspace_cnt':   'Revision Behavior',
    'r_burst_mean':               'Revision Behavior',
    'r_burst_std':                'Revision Behavior',
    'r_burst_median':             'Revision Behavior',
    'r_burst_max':                'Revision Behavior',
    'r_burst_first':              'Revision Behavior',

    # 8. Production Flow & Navigation
    'activity_Input_cnt':         'Production Flow & Navigation',
    'activity_Nonproduction_cnt': 'Production Flow & Navigation',
    'p_burst_count':              'Production Flow & Navigation',
    'p_burst_mean':               'Production Flow & Navigation',
    'p_burst_std':                'Production Flow & Navigation',
    'p_burst_median':             'Production Flow & Navigation',
    'p_burst_max':                'Production Flow & Navigation',
    'p_burst_first':              'Production Flow & Navigation',
    'p_burst_last':               'Production Flow & Navigation',
    'product_to_keys':            'Production Flow & Navigation',
    'cursor_position_max':        'Production Flow & Navigation',
    'cursor_position_mean':       'Production Flow & Navigation',
    'cursor_position_std':        'Production Flow & Navigation',
    'cursor_position_median':     'Production Flow & Navigation',
    'cursor_position_q1':         'Production Flow & Navigation',
    'cursor_position_q3':         'Production Flow & Navigation',
    'down_event_Leftclick_cnt':   'Production Flow & Navigation',
    'down_event_ArrowLeft_cnt':   'Production Flow & Navigation',
    'down_event_ArrowRight_cnt':  'Production Flow & Navigation',
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
