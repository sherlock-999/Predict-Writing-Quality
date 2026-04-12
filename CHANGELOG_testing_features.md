# Changelog: testing_features.py

## 2026-04-13 — Normalize Count-Based Features

**Issue:** The model is dominated by essay length because many behavioral features are raw counts that scale with essay size.

**Action:** Replaced raw count-based features with normalized versions (per-word). Added `session_duration_sec` as a new feature.

### Normalization Approach
- Count-based features normalized per-word: `feature / max(word_len_count, 1)`
- Dropped `up_event_*` features (mirrors of `down_event_*`)
- Kept distribution stats (mean, std, median, q1, q3, max, min)
- Kept anchor features (`word_len_sum`, `word_len_count`, `paragraph_count`)

### Features Normalized (per-word)

**Punctuation & Formatting (11 features):**
| Removed | Added |
|---------|-------|
| `down_event_comma_cnt` | `comma_per_word` |
| `text_change_comma_cnt` | `comma_text_per_word` |
| `up_event_comma_cnt` | (dropped) |
| `down_event_period_cnt` | `period_per_word` |
| `text_change_period_cnt` | `period_text_per_word` |
| `down_event_Enter_cnt` | `enter_per_word` |
| `text_change_newline_cnt` | `newline_per_word` |
| `down_event_Shift_cnt` | `shift_per_word` |
| `down_event_CapsLock_cnt` | `capslock_per_word` |
| `text_change_dash_cnt` | `dash_per_word` |
| `text_change_question_cnt` | `question_per_word` |
| `text_change_quote_cnt` | `quote_per_word` |

**Typing Speed (4 features):**
| Removed | Added |
|---------|-------|
| `down_event_q_cnt` | `q_keys_per_word` |
| `text_change_q_cnt` | `q_text_per_word` |
| `down_event_Space_cnt` | `space_per_word` |
| `text_change_space_cnt` | `space_text_per_word` |
| `up_event_q_cnt` | (dropped) |

**Pausing Behavior (5 features):**
| Removed | Added |
|---------|-------|
| `pauses_half_sec` | `pauses_half_per_word` |
| `pauses_1_sec` | `pauses_1_per_word` |
| `pauses_1_half_sec` | `pauses_1half_per_word` |
| `pauses_2_sec` | `pauses_2_per_word` |
| `pauses_3_sec` | `pauses_3_per_word` |

**Revision Behavior (2 features):**
| Removed | Added |
|---------|-------|
| `activity_RemoveCut_cnt` | `removecut_per_word` |
| `down_event_Backspace_cnt` | `backspace_per_word` |

**Production Flow (5 features):**
| Removed | Added |
|---------|-------|
| `activity_Input_cnt` | `input_per_word` |
| `activity_Nonproduction_cnt` | `nonproduction_per_word` |
| `p_burst_count` | `p_burst_per_word` |
| `down_event_Leftclick_cnt` | `leftclick_per_word` |
| `down_event_ArrowLeft_cnt` | `arrowleft_per_word` |
| `down_event_ArrowRight_cnt` | `arrowright_per_word` |

**Sentence Characteristics (1 feature):**
| Removed | Added |
|---------|-------|
| `sent_count` | `sent_per_word` |

### New Features Added

| Feature | Description |
|---------|-------------|
| `session_duration_sec` | Total session duration in seconds (`down_time_max - down_time_min`) |

### Dropped (Mirrors)

| Feature | Reason |
|---------|--------|
| `up_event_comma_cnt` | Mirror of `down_event_comma_cnt` |
| `up_event_q_cnt` | Mirror of `down_event_q_cnt` |
| `n_unique_up_event` | Mirror of `n_unique_down_event` |

---

## 2026-04-12 — Remove Duplicate Length Signals

**Issue:** The model was over-relying on multiple features that all represented essay length. This redundancy can lead to overfitting and reduced model generalization.

**Action:** Removed 5 features that were near-duplicates of essay length. Retained `word_len_sum` and `paragraph_count` as minimal anchor length features.

**Note:** This is a test file. `important_features.py` remains unchanged.

### Removed Features

| Feature | Reason |
|---------|--------|
| `sent_len_sum` | Total characters across sentences — essentially essay character length |
| `paragraph_len_sum` | Total characters across paragraphs — essentially essay character length |
| `word_count_max` | Max running word count — equals final essay word count |
| `cursor_position_max` | Furthest cursor position — proportional to essay character length |
| `paragraph_word_count_sum` | Total words across paragraphs — equals essay word count |

### Kept Anchor Features

| Feature | Justification |
|---------|---------------|
| `word_len_sum` | Total characters in final text — primary length anchor |
| `paragraph_count` | Total number of paragraphs — structural signal, not pure length |

### Not Removed (Per Guidance)

| Feature | Status |
|---------|--------|
| `word_len_count` | Word count signal — kept for now |
| All mean/median/std/q1/q3 variants | Capture distribution shape, not total size |

**Net Change:** 126 → 121 features
