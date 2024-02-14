""" from https://github.com/keithito/tacotron """
from text import cleaners
from text.symbols import symbols


# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


def text_to_sequence(text, symbols, cleaner_names):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = []
  symbol_to_id = {s: i for i, s in enumerate(symbols)}
  clean_text = _clean_text(text, cleaner_names)
  print(clean_text)
  print(f" length:{len(clean_text)}")
  for symbol in clean_text:
    if symbol not in symbol_to_id.keys():
      continue
    symbol_id = symbol_to_id[symbol]
    sequence += [symbol_id]
  print(f" length:{len(sequence)}")
  return sequence


def cleaned_text_to_sequence(cleaned_text, symbols):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  symbol_to_id = {s: i for i, s in enumerate(symbols)}
  sequence = [symbol_to_id[symbol] for symbol in cleaned_text if symbol in symbol_to_id.keys()]
  return sequence



from text.symbols import language_tone_start_map
def cleaned_text_to_sequence_vits2(cleaned_text, tones, language, symbols, languages):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    symbol_to_id = {s: i for i, s in enumerate(symbols)}
    language_id_map = {s: i for i, s in enumerate(languages)}
    phones = [symbol_to_id[symbol] for symbol in cleaned_text]
    tone_start = language_tone_start_map[language]
    tones = [i + tone_start for i in tones]
    lang_id = language_id_map[language]
    lang_ids = [lang_id for i in phones]
    return phones, tones, lang_ids


def sequence_to_text(sequence):
  '''Converts a sequence of IDs back to a string'''
  result = ''
  for symbol_id in sequence:
    s = _id_to_symbol[symbol_id]
    result += s
  return result


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text
