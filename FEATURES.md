# Feature Engineering

**226 features** computed per essay from raw keystroke logs (`v6_features.py`).

All alphabetic characters in the `text_change` column are anonymised as `q` by the competition organisers. Punctuation (`.`, `,`, `?`, `'`, `-`, `\n`) and control keys (`Space`, `Backspace`, `Enter`, etc.) are preserved exactly. Every feature below is computed from these anonymised logs.

---

## The Central Challenge

The competition provides no essay text directly — only a log of keystrokes. To predict writing quality we must answer: *what does a good writer's process look like, compared to a weaker one?*

Research in writing studies consistently shows that skilled writers produce longer texts, revise more strategically (rather than locally), maintain fluent typing with fewer extended pauses, and structure their essays with coherent paragraph organisation. Our feature engineering is grounded in these observations.

---

## Feature Categories

### 1. Word Characteristics (21 features)

**What they measure:** The length distribution of words in the final reconstructed essay, plus characteristics of words as the writer typed them.

The essay is reconstructed by replaying all keystrokes (inputs, deletions, replacements, pastes) in order. Word lengths are extracted from the result. We capture the full distribution: mean, max, Q3, first word, last word, and count.

Separately, *input words* — sequences of `q` characters in the `text_change` stream, ignoring replacement operations — approximate the words a writer typed directly (not pasted or auto-corrected). Their length distribution (mean, max, std, median, skew) reflects the writer's active vocabulary and typing granularity.

**Why it matters:** Longer words and a higher word count are consistently associated with higher essay scores. The skew of input word lengths captures whether a writer tends to type short function words vs. longer content words.

---

### 2. Sentence Characteristics (16 features)

**What they measure:** Length and word-count statistics over all sentences in the reconstructed essay (split on `.`, `?`, `!`).

We capture the full distribution of sentence character lengths and sentence word counts: mean, median, min, max, first, last, Q1, Q3. A derived feature `sent_per_word` normalises sentence count by essay length.

**Why it matters:** Sentence variety is a hallmark of sophisticated writing. Essays with uniformly short sentences score lower. The ratio of sentences to words reveals how densely the writer packs information — too high means telegraphic sentences; too low suggests run-on structures.

---

### 3. Paragraph Characteristics (17 features)

**What they measure:** Length and word-count statistics over paragraphs (split on `\n`), plus paragraph count.

**Why it matters:** Higher-scoring essays tend to have multiple, well-developed paragraphs of roughly balanced length. The `para_balance` feature (std of paragraph word counts) quantifies whether the writer distributes content evenly — a sign of deliberate structural planning.

---

### 4. Punctuation & Formatting (18 features)

**What they measure:** How often the writer uses each punctuation mark, normalised by essay word count. Derived from both the `down_event` column (what key was physically pressed) and the `text_change` column (what character appeared in the text). Where both sources exist, we include both (e.g., `comma_per_word` from key press, `comma_text_per_word` from text appearance).

Features cover: commas, periods, question marks, single quotes, double quotes, dashes, colons, semicolons, equals signs, slashes, backslashes, Enter key, newlines, Shift, and CapsLock.

**Why it matters:** Punctuation density is a direct signal of syntactic complexity. Writers who use semicolons, colons, and dashes are constructing more complex sentence structures. Normalising by word count removes the confound of essay length.

---

### 5. Typing Speed & Fluency (17 features)

**What they measure:** The tempo and rhythm of the writing session.

- `keys_per_second` — overall keystroke rate (input + deletion events / session duration)
- `action_time_*` — full distribution (min, mean, std, median, max, Q1, Q3, sum) of time between consecutive keystrokes
- `q_keys_per_word`, `q_text_per_word` — how many alphabetic keystrokes and characters appear per word
- `space_per_word`, `space_text_per_word` — space key usage normalised by word count
- `n_unique_activity`, `n_unique_text_change`, `n_unique_down_event`, `n_unique_up_event` — diversity of action types used

**Why it matters:** Fluent writers type more continuously. A high `action_time_std` relative to the mean indicates an irregular, hesitant rhythm. Key diversity (`n_unique_*`) reflects whether the writer used a variety of editing operations, which tends to correlate with active revision.

---

### 6. Pausing Behavior (10 features)

**What they measure:** Gaps between keystrokes (measured as the time between the `up_time` of one event and the `down_time` of the next, restricted to Input and Remove/Cut events).

The gap distribution is summarised with: `idle_largest_latency`, `idle_median_latency`, `idle_mean`, `idle_std`, `idle_total`. Pauses are also binned into five bands — 0.5–1 s, 1–1.5 s, 1.5–2 s, 2–3 s, and >3 s — and each count is normalised by word count.

**Why it matters:** Pausing is a proxy for cognitive load. Frequent long pauses indicate planning or difficulty. The *distribution* of pauses matters more than their total: a writer who pauses strategically at sentence boundaries is different from one who pauses randomly throughout. Normalising by word count controls for essay length.

---

### 7. Revision Behavior (8 features)

**What they measure:** How much and in what pattern the writer deleted text.

- `removecut_per_word` — total Remove/Cut events per word
- `backspace_per_word`, `delete_per_word` — specific deletion key rates
- `r_burst_*` — statistics over *revision bursts*: consecutive runs of Remove/Cut events (mean, std, median, max, first burst size)

**Why it matters:** Revision is a marker of metacognitive engagement. Writers who revise in longer bursts are making structural edits; writers who make many single-character deletions are correcting typos. The burst statistics distinguish these patterns. Note that *some* revision is good, but excessive revision without forward progress may signal difficulty.

---

### 8. Production Flow & Navigation (39 features)

**What they measure:** How the writer moved through the document and how efficiently they produced text.

- `input_per_word`, `replace_per_word`, `paste_per_word`, `nonproduction_per_word` — activity type breakdown normalised by word count
- `p_burst_*` — production burst statistics: consecutive runs of Input/Remove/Cut events without a gap > 2 s (mean, std, median, max, first, last, count per word)
- `product_to_keys` — ratio of final essay length to total keystrokes (efficiency: how many characters survive per keystroke)
- `session_duration_sec` — total session length
- `cursor_position_*` — full distribution of cursor positions (min, mean, std, median, Q1, Q3, max)
- `down_time_*`, `up_time_*` — full timestamp distributions (min, mean, std, median, Q1, Q3, max)
- `leftclick_per_word`, `arrowleft_per_word`, `arrowright_per_word`, `arrowdown_per_word`, `arrowup_per_word`, `unidentified_per_word` — navigation key rates

**Why it matters:** `product_to_keys` is a direct efficiency measure — a value near 1.0 means almost every keystroke contributed to the final essay; a low value indicates heavy deletion. Production burst length reflects sustained fluent writing periods. Cursor navigation statistics reveal whether the writer moved around the document to revise earlier sections (large cursor range) or only wrote linearly at the end.

---

### 9. Up-Event Counts (16 features)

**What they measure:** Raw counts of each key-release (`up_event`) for the 16 most frequent keys: alphabetic (`q`), Space, Backspace, Shift, ArrowRight, Leftclick, ArrowLeft, period, comma, ArrowDown, ArrowUp, Enter, CapsLock, single quote, Delete, Unidentified.

**Why it matters:** The `up_event` column is independent of `down_event` (key press) — both are captured separately in the log. Up-event counts are not redundant with down-event counts because key-repeat behaviour and timing can differ. These raw counts complement the normalised per-word features by preserving absolute volume information.

---

### 10. Raw Activity / Event Counts (13 features)

**What they measure:** Absolute counts of less-common activities and events that are informative but not common enough to dominate the per-word normalised features:

- `activity_Replace_cnt`, `activity_Paste_cnt` — bulk edit operations
- `text_change_doublequote_cnt`, `text_change_semicolon_cnt`, `text_change_equals_cnt`, `text_change_slash_cnt`, `text_change_backslash_cnt`, `text_change_colon_cnt` — rarer punctuation in text
- `down_event_ArrowDown_cnt`, `down_event_ArrowUp_cnt`, `down_event_singlequote_cnt`, `down_event_Delete_cnt`, `down_event_Unidentified_cnt` — rarer key events

**Why it matters:** These preserve absolute frequency information (not normalised), which captures the scale of operations. A writer who pastes 10 times is behaviourally different from one who pastes once, even if their word counts differ.

---

### 11. Readability & Structural Consistency (8 features)

**What they measure:** Objective readability scores and consistency metrics computed on the reconstructed essay text.

- `ari_score` — Automated Readability Index: `4.71 × (chars/words) + 0.5 × (words/sentences) − 21.43`. Higher = more complex vocabulary and longer sentences.
- `coleman_liau_score` — Coleman-Liau Index: based on characters per 100 words and sentences per 100 words. Predicts US grade level.
- `sent_len_std`, `sent_len_cv` — sentence length standard deviation and coefficient of variation: how consistent is sentence length?
- `para_len_std`, `para_len_cv` — same for paragraphs
- `para_balance` — std of word counts across paragraphs: are paragraphs evenly developed?
- `words_per_minute` — essay word count divided by session duration in minutes

**Why it matters:** Higher-scoring essays tend to score higher on readability indices, reflecting more advanced vocabulary and sentence structure. Sentence length CV captures stylistic variety: skilled writers vary sentence length intentionally, but excessive variation may indicate inconsistency. Words per minute gives a direct measure of writing fluency.

---

### 12. Revision Timing (5 features)

**What they measure:** *When* during the session the writer made deletions — whether revisions were concentrated early, mid, or late in the session.

The session is divided into thirds by wall-clock time. `revision_ratio_early`, `revision_ratio_mid`, `revision_ratio_late` give the fraction of all Remove/Cut events that fall in each third. `revision_timing_mean` and `revision_timing_std` summarise the normalised timing distribution directly.

**Why it matters:** Writing research distinguishes *planning-driven* writers (who revise early or in planning phases) from *reactive* writers (who revise only at the end). Early revision ratios may reflect outlining or restructuring behaviour; late revision may indicate proofreading. The temporal distribution of revision is a behavioural fingerprint not captured by raw revision counts.

---

### 13. Text TF-IDF SVD (20 features)

**What they measure:** 20 latent semantic dimensions of the reconstructed essay text, derived from a TF-IDF vectoriser followed by Truncated SVD.

**Pipeline:** The essay text is vectorised with `TfidfVectorizer(analyzer='char', ngram_range=(2, 4), sublinear_tf=True)`, capturing character-level 2–4-gram patterns (punctuation rhythm, suffix patterns, capitalization). The resulting sparse matrix is compressed to 20 dimensions with `TruncatedSVD`.

The pipeline is fitted on the training corpus only and applied (transform-only) to test data.

**Why it matters:** Character n-grams capture writing style at a level that word-level features miss. Common 3-grams like `", "` (comma-space) or `"ed "` (past-tense suffix) reflect punctuation habits and vocabulary choices. The 20 SVD components encode the dominant stylistic axes across the training essays.

---

### 14. Event TF-IDF SVD (20 features)

**What they measure:** 20 latent dimensions of the keystroke sequence itself, treating the `down_event` column as a document of tokens.

**Pipeline:** For each essay, the `down_event` values are joined into a space-separated string. Punctuation keys (`.`, `,`, `'`) are remapped to word tokens (`period`, `comma`, `singlequote`) so the word tokeniser preserves them. The string is vectorised with `TfidfVectorizer(analyzer='word', ngram_range=(1, 5), sublinear_tf=True)`, capturing 1–5-gram patterns in the keystroke sequence. Truncated SVD reduces to 20 dimensions.

Example n-grams captured: `"Shift q"` (capitalisation), `"Backspace Backspace Backspace"` (burst deletion), `"ArrowLeft ArrowLeft q"` (cursor-back-and-retype), `"Enter Enter q"` (paragraph break then immediate typing).

**Why it matters:** This is a bag-of-n-grams over the *process*, not the product. It captures rhythmic keystroke patterns that aggregate features cannot. A writer who habitually uses `ArrowLeft` to correct mid-word has a different n-gram signature from one who uses `Backspace`. These patterns are invisible to the count-based features above.

---

## Feature Engineering Pipeline

```
raw keystroke log (train_logs.csv)
        │
        ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Per-essay groupby (groupby 'id', sort by 'event_id')   │
  │                                                         │
  │  1. Count features          (activity / text_change /   │
  │                              down_event / up_event)     │
  │  2. Input word features     (q-run lengths in           │
  │                              text_change stream)        │
  │  3. Timing features         (action_time, down_time,    │
  │                              up_time, cursor_position,  │
  │                              word_count distributions)  │
  │  4. Idle / pause features   (inter-keystroke gaps)      │
  │  5. Production burst feats  (p_burst statistics)        │
  │  6. Revision burst feats    (r_burst statistics)        │
  │  7. Essay reconstruction    (replay all edits → text)   │
  │  8. Word features           (length distribution)       │
  │  9. Sentence features       (length / word-count dist)  │
  │ 10. Paragraph features      (length / word-count dist)  │
  │ 11. Efficiency features     (product_to_keys, KPS)      │
  │ 12. Session duration        (from timing features)      │
  │ 13. Per-word normalisation  (counts ÷ word_len_count)   │
  │ 14. Readability features    (ARI, CLI, consistency)     │
  │ 15. Revision timing         (early / mid / late ratios) │
  └─────────────────────────────────────────────────────────┘
        │                              │
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

Both TF-IDF pipelines are fitted on the training corpus and saved as `tfidf_svd.pkl` and `event_tfidf_svd.pkl`. At inference time they are loaded and applied with `transform()` only — no refitting on test data.
