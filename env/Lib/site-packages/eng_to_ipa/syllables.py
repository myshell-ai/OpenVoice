import re
import os
import json
from eng_to_ipa import transcribe


with open(os.path.join(os.path.abspath(os.path.dirname(__file__)),
                       'resources','phones.json'), "r") as phones_json:
    PHONES = json.load(phones_json)

# list of adjacent vowel symbols that constitute separate nuclei
hiatus = [["er", "iy"], ["iy", "ow"], ["uw", "ow"], ["iy", "ah"], ["iy", "ey"], ["uw", "eh"], ["er", "eh"]]


def cmu_syllable_count(word):
    """count syllables based on CMU transcription"""
    word = re.sub("\d", "", word).split(' ')
    if "__IGNORE__" in word[0]:
        return 0
    else:
        nuclei = 0
        for i, sym in enumerate(word):
            prev_phone = PHONES[word[i-1]]
            prev_sym = word[i-1]
            if PHONES[sym] == 'vowel':
                if i > 0 and not prev_phone == 'vowel' or i == 0:
                    nuclei += 1
                elif [prev_sym, sym] in hiatus:
                    nuclei += 1
        return nuclei


def syllable_count(word: str, db_type="sql"):
    """transcribes a regular word to CMU to fetch syllable count"""
    if len(word.split()) > 1:
        return [syllable_count(w) for w in word.split()]
    word = transcribe.get_cmu([transcribe.preprocess(word)], db_type=db_type)
    return cmu_syllable_count(word[0][0])
