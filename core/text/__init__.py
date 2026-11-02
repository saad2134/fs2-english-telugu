"""
Text processing and phonemization interface for FastSpeech 2.
Supports English (ARPAbet / G2P) and Telugu (Indic Akshara / Phonetic G2P).
"""

import re
try:
    from core.text import cleaners
    from core.text.symbols import symbols, symbol_to_id, id_to_symbol
    from core.text.telugu import telugu_to_phonemes, telugu_text_to_sequence
except ImportError:
    from text import cleaners
    from text.symbols import symbols, symbol_to_id, id_to_symbol
    from text.telugu import telugu_to_phonemes, telugu_text_to_sequence

_symbol_to_id = symbol_to_id
_id_to_symbol = id_to_symbol

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")


def text_to_sequence(text, cleaner_names, language="english"):
    """
    Converts a string of text to a sequence of symbol IDs.
    
    Args:
        text (str): Input text (English or Telugu).
        cleaner_names (list): Cleaner function names.
        language (str): 'english' or 'telugu'.
    """
    if language.lower() == "telugu" or any('\u0c00' <= char <= '\u0c7f' for char in text):
        return telugu_text_to_sequence(text, _symbol_to_id)
        
    sequence = []
    # Check for curly braces and treat their contents as ARPAbet:
    while len(text):
        m = _curly_re.match(text)
        if not m:
            sequence += _symbols_to_sequence(_clean_text(text, cleaner_names))
            break
        sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
        sequence += _arpabet_to_sequence(m.group(2))
        text = m.group(3)

    return sequence


def sequence_to_text(sequence):
    """Converts a sequence of IDs back to a string"""
    result = ""
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            if len(s) > 1 and s[0] == "@":
                s = "{%s}" % s[1:]
            result += s
    return result.replace("}{", " ")


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name, None)
        if cleaner is not None:
            text = cleaner(text)
    return text


def _symbols_to_sequence(symbols_list):
    return [_symbol_to_id[s] for s in symbols_list if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
    seq = []
    for s in text.split():
        if "@" + s in _symbol_to_id:
            seq.append(_symbol_to_id["@" + s])
        elif s in _symbol_to_id:
            seq.append(_symbol_to_id[s])
        elif "@" + s.upper() in _symbol_to_id:
            seq.append(_symbol_to_id["@" + s.upper()])
        elif "@sp" in _symbol_to_id:
            seq.append(_symbol_to_id["@sp"])
    return seq


def _should_keep_symbol(s):
    return s in _symbol_to_id and s != "_" and s != "~"
