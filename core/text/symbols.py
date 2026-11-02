"""
Defines the unified set of symbols used in text input to the FastSpeech 2 models
for both English (ARPAbet) and Telugu (Indic Phonetic Inventory).
"""

try:
    from core.text import cmudict
    from core.text.telugu import TELUGU_PHONEMES
except ImportError:
    from text import cmudict
    from text.telugu import TELUGU_PHONEMES

_pad = "_"
_punctuation = "!'(),.:;? -~"
_special = "-#"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_silences = ["@sp", "@spn", "@sil", "sp", "sil"]

# Prepend "@" to ARPAbet symbols for English uniqueness:
_arpabet = ["@" + s for s in cmudict.valid_symbols]

# Telugu phonemes with @te_ prefix and standard form
_telugu_symbols = [f"@te_{p}" for p in TELUGU_PHONEMES] + TELUGU_PHONEMES

# Export all symbols uniquely while preserving ordering:
raw_symbols = (
    [_pad]
    + list(_special)
    + list(_punctuation)
    + list(_letters)
    + _arpabet
    + _telugu_symbols
    + _silences
)

symbols = []
for s in raw_symbols:
    if s not in symbols:
        symbols.append(s)

# Symbol to ID and ID to symbol mappings
symbol_to_id = {s: i for i, s in enumerate(symbols)}
id_to_symbol = {i: s for i, s in enumerate(symbols)}
