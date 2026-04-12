"""
Efficiency Features
===================
Two ratios that measure how effectively the writer converted keystrokes
into actual essay text.

Processing order inside compute_features():
    Step 7 : efficiency_features(essay, essay_text)
             (requires the reconstructed essay text from Step 3)
"""

import pandas as pd


def efficiency_features(essay: pd.DataFrame, essay_text: str) -> dict:
    """
    Compute product-to-keys ratio and typing speed in keys per second.

    product_to_keys
        Net characters in the final essay ÷ total Input/Remove events.
        Higher = more of each keystroke survived into the final text
        (low revision overhead).

    keys_per_second
        Total Input/Remove events ÷ active session duration in seconds.
        Measures gross typing speed regardless of how much was deleted.

    Parameters
    ----------
    essay : pd.DataFrame
        All rows for one essay, sorted by event_id.
    essay_text : str
        Reconstructed final essay text (from essay_reconstruction.py).

    Returns
    -------
    dict
        product_to_keys, keys_per_second
    """
    keys_pressed    = int(essay['activity'].isin(['Input', 'Remove/Cut']).sum())
    session_seconds = (essay['up_time'].max() - essay['down_time'].min()) / 1000

    return {
        'product_to_keys': len(essay_text) / max(keys_pressed, 1),
        'keys_per_second': keys_pressed    / max(session_seconds, 1),
    }
