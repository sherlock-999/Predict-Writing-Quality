# Feature Engineering

**226 features** computed per essay from raw keystroke logs (`v6_features.py`), organised into 14 categories.

> **Note on anonymisation:** All alphabetic characters in `text_change` are replaced with `q` by the competition organisers. Punctuation (`.`, `,`, `?`, `'`, `-`, `\n`) and control keys (`Space`, `Backspace`, `Enter`, etc.) are preserved exactly.

---

## The Core Problem

The competition provides no essay text — only a log of keystrokes. To predict writing quality we must ask: **what does a good writer's process look like?**

Research in writing studies shows skilled writers:
- Produce **longer, more complex** texts
- Revise **strategically** (structural edits) rather than reactively (typo fixes)
- Type **more fluently** with fewer extended pauses
- **Distribute content** across well-developed paragraphs

Every feature category below is grounded in one of these observations.

---

## Feature Categories at a Glance

| # | Category | Features | What it captures |
|---|----------|----------|-----------------|
| 1 | Word Characteristics | 21 | Final essay word lengths; typed word lengths |
| 2 | Sentence Characteristics | 16 | Sentence length distribution and variety |
| 3 | Paragraph Characteristics | 17 | Paragraph length, count, balance |
| 4 | Punctuation & Formatting | 18 | Punctuation density by type, normalised by word count |
| 5 | Typing Speed & Fluency | 17 | Keystroke rate, inter-key timing, action diversity |
| 6 | Pausing Behaviour | 10 | Idle gap distribution; pause frequency by duration band |
| 7 | Revision Behaviour | 8 | Deletion rate; burst size (typo fix vs. structural edit) |
| 8 | Production Flow & Navigation | 39 | Efficiency, burst lengths, cursor movement, session timeline |
| 9 | Up-Event Counts | 16 | Raw key-release counts for 16 most frequent keys |
| 10 | Raw Activity / Event Counts | 13 | Absolute counts of rarer operations (paste, replace, rare punctuation) |
| 11 | Readability & Structural Consistency | 8 | ARI, Coleman-Liau, sentence/paragraph consistency |
| 12 | Revision Timing | 5 | When during the session deletions occurred (early / mid / late) |
| 13 | Text TF-IDF SVD | 20 | Latent style axes from char 2–4-gram TF-IDF on reconstructed essay |
| 14 | Event TF-IDF SVD | 20 | Latent keystroke-pattern axes from word 1–5-gram TF-IDF on key sequence |
| | **Total** | **226** | |

---

## Category Details

### 1. Word Characteristics (21 features)

**Source:** Reconstructed essay text + `text_change` stream.

| Feature(s) | Description |
|------------|-------------|
| `word_len_mean`, `word_len_max`, `word_len_q3` | Average, max, and 75th-percentile word length in the final essay |
| `word_len_first`, `word_len_last` | Length of the first and last word |
| `word_len_count` | Total word count (essay length proxy) |
| `input_word_mean`, `_max`, `_std`, `_median`, `_skew` | Length statistics of words typed directly (runs of `q` in `text_change`) |

**Why it matters:**
- Longer words → richer vocabulary → higher scores
- `input_word_skew` captures whether the writer leans toward short function words vs. longer content words
- Word count is one of the strongest single predictors of score

---

### 2. Sentence Characteristics (16 features)

**Source:** Reconstructed essay, split on `.`, `?`, `!`.

| Feature(s) | Description |
|------------|-------------|
| `sent_char_mean`, `_median`, `_min`, `_max`, `_q1`, `_q3` | Distribution of sentence character lengths |
| `sent_word_mean`, `_median`, `_min`, `_max` | Distribution of sentence word counts |
| `sent_count` | Total number of sentences |
| `sent_per_word` | Sentences per word (sentence density) |

**Why it matters:**
- Uniformly short sentences → lower scores; variety in sentence length → higher scores
- `sent_per_word` flags run-on sentences (too low) or telegraphic writing (too high)

---

### 3. Paragraph Characteristics (17 features)

**Source:** Reconstructed essay, split on `\n`.

| Feature(s) | Description |
|------------|-------------|
| `para_char_mean`, `_median`, `_min`, `_max`, `_q1`, `_q3` | Distribution of paragraph character lengths |
| `para_word_mean`, `_median`, `_min`, `_max` | Distribution of paragraph word counts |
| `para_count` | Number of paragraphs |
| `para_balance` | Std of paragraph word counts — how evenly developed each paragraph is |

**Why it matters:**
- Multiple well-developed paragraphs → deliberate structural planning → higher scores
- `para_balance` distinguishes writers who plan structure from those who dump everything into one block

---

### 4. Punctuation & Formatting (18 features)

**Source:** `down_event` column (key pressed) and `text_change` column (character in text), normalised by word count.

| Feature(s) | Description |
|------------|-------------|
| `comma_per_word`, `comma_text_per_word` | Comma rate (from key press / from text appearance) |
| `period_per_word` | Period rate |
| `question_per_word` | Question mark rate |
| `singlequote_per_word`, `doublequote_text_per_word` | Quotation mark rates |
| `dash_per_word` | Dash rate |
| `colon_text_per_word`, `semicolon_text_per_word` | Colon and semicolon rates |
| `enter_per_word`, `newline_per_word` | Enter key and newline character rates |
| `shift_per_word`, `capslock_per_word` | Capitalisation key rates |
| `equals_text_per_word`, `slash_text_per_word`, `backslash_text_per_word` | Rare punctuation rates |

**Why it matters:**
- Semicolons, colons, and dashes signal complex sentence structures
- Normalising by word count removes the confound of essay length

---

### 5. Typing Speed & Fluency (17 features)

**Source:** Event timestamps.

| Feature(s) | Description |
|------------|-------------|
| `keys_per_second` | Overall keystroke rate (all events / session duration) |
| `action_time_mean`, `_std`, `_median`, `_min`, `_max`, `_q1`, `_q3`, `_sum` | Full distribution of time between consecutive keystrokes |
| `q_keys_per_word`, `q_text_per_word` | Alphabetic keystrokes and characters per final word |
| `space_per_word`, `space_text_per_word` | Space key rate |
| `n_unique_activity`, `n_unique_text_change`, `n_unique_down_event`, `n_unique_up_event` | Diversity of action types used |

**Why it matters:**
- Fluent writers type continuously → lower `action_time_std` relative to mean
- High `n_unique_*` → varied editing operations → active revision

---

### 6. Pausing Behaviour (10 features)

**Source:** Gaps between `up_time` of one event and `down_time` of the next (Input and Remove/Cut events only).

| Feature(s) | Description |
|------------|-------------|
| `idle_largest_latency` | Longest single pause in the session |
| `idle_median_latency`, `idle_mean`, `idle_std`, `idle_total` | Summary statistics of all pause durations |
| `pause_0.5_1_per_word` | Pauses 0.5–1 s per word |
| `pause_1_1.5_per_word` | Pauses 1–1.5 s per word |
| `pause_1.5_2_per_word` | Pauses 1.5–2 s per word |
| `pause_2_3_per_word` | Pauses 2–3 s per word |
| `pause_3plus_per_word` | Pauses > 3 s per word |

**Why it matters:**
- Frequent long pauses → cognitive load or difficulty
- The **distribution** of pauses matters: strategic pausing at sentence boundaries differs from random hesitation
- Normalising by word count controls for essay length

---

### 7. Revision Behaviour (8 features)

**Source:** Remove/Cut and Backspace/Delete events.

| Feature(s) | Description |
|------------|-------------|
| `removecut_per_word` | Total Remove/Cut events per word |
| `backspace_per_word` | Backspace key rate per word |
| `delete_per_word` | Delete key rate per word |
| `r_burst_mean`, `r_burst_std`, `r_burst_median`, `r_burst_max` | Statistics of revision burst lengths (consecutive deletions) |
| `r_burst_first` | Size of the first revision burst |

**Why it matters:**
- Some revision is a positive signal (metacognitive engagement); excessive deletion without progress is negative
- **Burst size distinguishes edit types:** single-character bursts = typo fixes; long bursts = structural rewrites
- `r_burst_max` captures the largest single revision event in the session

---

### 8. Production Flow & Navigation (39 features)

**Source:** Event types, timestamps, cursor positions, word counts.

| Feature(s) | Description |
|------------|-------------|
| `input_per_word`, `replace_per_word`, `paste_per_word`, `nonproduction_per_word` | Activity type breakdown per word |
| `p_burst_mean`, `_std`, `_median`, `_max`, `_first`, `_last`, `_count_per_word` | Production burst statistics (consecutive input/edit events without a >2 s gap) |
| `product_to_keys` | Final essay length ÷ total keystrokes (efficiency: characters surviving per keystroke) |
| `session_duration_sec` | Total session length in seconds |
| `cursor_position_mean`, `_std`, `_median`, `_min`, `_max`, `_q1`, `_q3` | Distribution of cursor positions throughout the session |
| `down_time_mean`, `_std`, `_median`, `_min`, `_max`, `_q1`, `_q3` | Distribution of key-down timestamps |
| `up_time_mean`, `_std`, `_median`, `_min`, `_max`, `_q1`, `_q3` | Distribution of key-up timestamps |
| `leftclick_per_word`, `arrowleft_per_word`, `arrowright_per_word` | Horizontal navigation rates |
| `arrowdown_per_word`, `arrowup_per_word` | Vertical navigation rates |
| `unidentified_per_word` | Rate of unrecognised key events |

**Why it matters:**
- `product_to_keys` near 1.0 → almost every keystroke survived; low value → heavy deletion
- Long production bursts → sustained fluent writing
- Large cursor position range → writer moved back to revise earlier sections (non-linear writing)

---

### 9. Up-Event Counts (16 features)

**Source:** `up_event` column (key-release events).

| Feature(s) | Description |
|------------|-------------|
| `up_q_cnt` | Key-release count for alphabetic keys (`q`) |
| `up_space_cnt`, `up_backspace_cnt`, `up_shift_cnt` | Space, Backspace, Shift releases |
| `up_arrowright_cnt`, `up_arrowleft_cnt`, `up_arrowdown_cnt`, `up_arrowup_cnt` | Arrow key releases |
| `up_leftclick_cnt` | Mouse left-click releases |
| `up_period_cnt`, `up_comma_cnt`, `up_singlequote_cnt` | Punctuation key releases |
| `up_enter_cnt`, `up_capslock_cnt` | Enter and CapsLock releases |
| `up_delete_cnt`, `up_unidentified_cnt` | Delete and unidentified key releases |

**Why it matters:**
- `up_event` and `down_event` are captured independently — key-repeat and timing differ between press and release
- These **absolute counts** complement the normalised per-word features by preserving volume information

---

### 10. Raw Activity / Event Counts (13 features)

**Source:** `activity`, `text_change`, `down_event` columns — raw counts, not normalised.

| Feature(s) | Description |
|------------|-------------|
| `activity_Replace_cnt` | Total Replace operations |
| `activity_Paste_cnt` | Total Paste operations |
| `text_change_doublequote_cnt`, `text_change_semicolon_cnt` | Double-quote and semicolon appearances in text |
| `text_change_equals_cnt`, `text_change_slash_cnt`, `text_change_backslash_cnt`, `text_change_colon_cnt` | Rarer punctuation in text |
| `down_event_ArrowDown_cnt`, `down_event_ArrowUp_cnt` | Vertical arrow key press counts |
| `down_event_singlequote_cnt`, `down_event_Delete_cnt`, `down_event_Unidentified_cnt` | Single-quote, Delete, and unidentified key press counts |

**Why it matters:**
- Preserves **absolute frequency** rather than normalised rate — a writer who pastes 10 times differs from one who pastes once, even if word counts are similar
- Rarer events (semicolons, colons) are often too infrequent to normalise reliably; raw counts work better

---

### 11. Readability & Structural Consistency (8 features)

**Source:** Reconstructed essay text.

| Feature(s) | Description |
|------------|-------------|
| `ari_score` | Automated Readability Index: `4.71×(chars/words) + 0.5×(words/sentences) − 21.43` |
| `coleman_liau_score` | Coleman-Liau Index: based on chars per 100 words and sentences per 100 words |
| `sent_len_std`, `sent_len_cv` | Sentence length std and coefficient of variation |
| `para_len_std`, `para_len_cv` | Paragraph length std and coefficient of variation |
| `para_balance` | Std of word counts across paragraphs |
| `words_per_minute` | Essay word count ÷ session duration (minutes) |

**Why it matters:**
- ARI and Coleman-Liau reflect vocabulary and syntactic complexity — both correlate with score
- CV features capture **stylistic variety**: skilled writers vary sentence length intentionally; excessive uniformity signals lower quality
- `words_per_minute` is a direct fluency measure independent of raw speed

---

### 12. Revision Timing (5 features)

**Source:** Timestamps of Remove/Cut events, session divided into thirds.

| Feature(s) | Description |
|------------|-------------|
| `revision_ratio_early` | Fraction of all deletions in the first third of the session |
| `revision_ratio_mid` | Fraction of all deletions in the middle third |
| `revision_ratio_late` | Fraction of all deletions in the final third |
| `revision_timing_mean` | Mean normalised time of deletion events (0 = session start, 1 = end) |
| `revision_timing_std` | Spread of deletion timing across the session |

**Why it matters:**
- **Early revision** → planning-driven writer (restructuring before committing)
- **Late revision** → reactive writer (proofreading at the end)
- **Spread (`revision_timing_std`)** → revision distributed throughout = more metacognitive engagement
- These patterns are invisible to raw deletion counts

---

### 13. Text TF-IDF SVD (20 features)

**Source:** Reconstructed essay text → TF-IDF → SVD.

**Pipeline:**
1. Reconstruct full essay text by replaying all keystroke edits
2. Vectorise with `TfidfVectorizer(analyzer='char', ngram_range=(2,4), sublinear_tf=True)`
3. Compress to 20 dimensions with `TruncatedSVD`
4. Features: `tfidf_svd_0` … `tfidf_svd_19`

| Step | What it captures |
|------|-----------------|
| Char 2-grams | Punctuation pairs (`, `, `. `), common endings (`e `, `s `) |
| Char 3-grams | Suffix patterns (`ing`, `ed `, `ion`), capitalisation contexts |
| Char 4-grams | Longer stylistic patterns, vocabulary fingerprints |
| SVD 20 dims | 20 dominant stylistic axes across all training essays |

**Why it matters:**
- Character n-grams capture writing **style** at a level word features miss
- The 20 SVD components encode the main directions of stylistic variation in the training corpus
- Fitted on training data only; applied transform-only at inference — no data leakage

---

### 14. Event TF-IDF SVD (20 features)

**Source:** `down_event` column → treat as token sequence → TF-IDF → SVD.

**Pipeline:**
1. Join all `down_event` values per essay into a space-separated string
2. Remap punctuation keys: `.` → `period`, `,` → `comma`, `'` → `singlequote` (so the word tokeniser preserves them as distinct tokens)
3. Vectorise with `TfidfVectorizer(analyzer='word', ngram_range=(1,5), sublinear_tf=True)`
4. Compress to 20 dimensions with `TruncatedSVD`
5. Features: `event_tfidf_svd_0` … `event_tfidf_svd_19`

| Example n-gram | What it represents |
|---------------|--------------------|
| `Shift q` | Capitalisation — typing an uppercase letter |
| `Backspace Backspace Backspace` | Burst deletion — 3 consecutive backspaces |
| `ArrowLeft ArrowLeft q` | Cursor-back-and-retype pattern |
| `Enter Enter q` | Paragraph break followed immediately by typing |
| `period q` | Starting a new sentence without a space pause |

**Why it matters:**
- This is a bag-of-n-grams over the **process**, not the product
- Captures rhythmic keystroke patterns invisible to count features
- A writer who uses `ArrowLeft` to correct mid-word has a different n-gram signature than one who uses `Backspace` — the aggregate counts are the same, but the sequences differ

---

## Full Feature Engineering Pipeline

```
raw keystroke log (train_logs.csv)
        │
        ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Per-essay groupby (groupby 'id', sort by 'event_id')   │
  │                                                         │
  │   1. Count features        activity / text_change /     │
  │                            down_event / up_event cols   │
  │   2. Input word features   q-run lengths in             │
  │                            text_change stream           │
  │   3. Timing features       action_time, down_time,      │
  │                            up_time, cursor_position,    │
  │                            word_count distributions     │
  │   4. Idle / pause features inter-keystroke gaps         │
  │   5. Production bursts     p_burst statistics           │
  │   6. Revision bursts       r_burst statistics           │
  │   7. Essay reconstruction  replay all edits → text      │
  │   8. Word features         length distribution          │
  │   9. Sentence features     length / word-count dist     │
  │  10. Paragraph features    length / word-count dist     │
  │  11. Efficiency features   product_to_keys, KPS         │
  │  12. Session duration      from timing features         │
  │  13. Per-word normalise    counts ÷ word_len_count      │
  │  14. Readability features  ARI, CLI, consistency        │
  │  15. Revision timing       early / mid / late ratios    │
  └─────────────────────────────────────────────────────────┘
        │                              │
        ▼                              ▼
  text corpus                   event sequences
  (reconstructed essays)        (down_event tokens per essay)
        │                              │
  TfidfVectorizer                TfidfVectorizer
  char 2–4-gram                  word 1–5-gram
  sublinear_tf=True              sublinear_tf=True
        │                              │
  TruncatedSVD(20)               TruncatedSVD(20)
        │                              │
  tfidf_svd_0…19                 event_tfidf_svd_0…19
        │                              │
        └──────────────┬───────────────┘
                       ▼
             226-feature DataFrame
             (one row per essay)
```

Both TF-IDF pipelines are **fitted on training data only** and saved as `tfidf_svd.pkl` and `event_tfidf_svd.pkl`. At inference time they are loaded and applied with `transform()` — no refitting on test data.
