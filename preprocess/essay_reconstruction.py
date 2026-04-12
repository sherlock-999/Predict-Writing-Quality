"""
Essay Reconstruction
====================
Replays all keystroke events in order to recover the final essay text.

The text is stored as a mutable string and updated event by event:
  - Input  : insert characters at the current cursor position
  - Remove/Cut : delete characters at the current cursor position
  - Replace    : swap an old substring for a new one (autocorrect / find-replace)
  - Paste      : insert a pasted block at the current cursor position
  - Move From  : move a text range from one position to another
  - Nonproduction (navigation, Shift, Arrow, etc.) : no text change, skip
"""

import pandas as pd


def reconstruct_essay(essay: pd.DataFrame) -> str:
    """
    Replay keystroke events to recover the final essay text.

    Parameters
    ----------
    essay : pd.DataFrame
        All rows for one essay, already sorted by event_id.
        Nonproduction events should be excluded before calling this function
        (they carry 'NoChange' or NaN in text_change and don't affect text).

    Returns
    -------
    str
        The reconstructed essay text.
    """
    text = ""

    for _, row in essay[['activity', 'cursor_position', 'text_change']].iterrows():
        act = row['activity']
        cur = row['cursor_position']
        tc  = row['text_change']

        if act == 'Replace':
            # text_change format: "old_text => new_text"
            parts = tc.split(' => ')
            if len(parts) == 2:
                old, new = parts
                text = text[:cur - len(new)] + new + text[cur - len(new) + len(old):]

        elif act == 'Paste':
            text = text[:cur - len(tc)] + tc + text[cur - len(tc):]

        elif act == 'Remove/Cut':
            text = text[:cur] + text[cur + len(tc):]

        elif act.startswith('Move From'):
            # Move From [x1, y1] To [x2, y2] — reorder a text slice
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
                pass  # malformed Move event — leave text unchanged

        else:
            # Input: insert tc at position (cur - len(tc))
            text = text[:cur - len(tc)] + tc + text[cur - len(tc):]

    return text
