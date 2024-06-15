# -*- coding: utf-8 -*-
import re
from os.path import join, abspath, dirname
import eng_to_ipa.stress as stress
from collections import defaultdict


def mode_type(mode_in):
    """In the case of "sql", this will return an sqlite cursor.
       In the case of "json", this will return a json dictionary of the data."""
    if mode_in.lower() == "sql":
        import sqlite3
        conn = sqlite3.connect(join(abspath(dirname(__file__)), "./resources/CMU_dict.db"))
        return conn.cursor()
    elif mode_in.lower() == "json":
        import json
        json_file = open(join(abspath(dirname(__file__)), "../eng_to_ipa/resources/CMU_dict.json"), encoding="UTF-8")
        return json.load(json_file)


def preprocess(words):
    """Returns a string of words stripped of punctuation"""
    punct_str = '!"#$%&\'()*+,-./:;<=>/?@[\\]^_`{|}~«» '
    return ' '.join([w.strip(punct_str).lower() for w in words.split()])


def preserve_punc(words):
    """converts words to IPA and finds punctuation before and after the word."""
    words_preserved = []
    for w in words.split():
        punct_list = ["", preprocess(w), ""]
        before = re.search("^([^A-Za-z0-9]+)[A-Za-z]", w)
        after = re.search("[A-Za-z]([^A-Za-z0-9]+)$", w)
        if before:
            punct_list[0] = str(before.group(1))
        if after:
            punct_list[2] = str(after.group(1))
        words_preserved.append(punct_list)
    return words_preserved


def apply_punct(triple, as_str=False):
    """places surrounding punctuation back on center on a list of preserve_punc triples"""
    if type(triple[0]) == list:
        for i, t in enumerate(triple):
            triple[i] = str(''.join(triple[i]))
        if as_str:
            return ' '.join(triple)
        return triple
    if as_str:
        return str(''.join(t for t in triple))
    return [''.join(t for t in triple)]


def _punct_replace_word(original, transcription):
    """Get the IPA transcription of word with the original punctuation marks"""
    for i, trans_list in enumerate(transcription):
        for j, item in enumerate(trans_list):
            triple = [original[i][0]] + [item] + [original[i][2]]
            transcription[i][j] = apply_punct(triple, as_str=True)
    return transcription


def fetch_words(words_in, db_type="sql"):
    """fetches a list of words from the database"""
    asset = mode_type(db_type)
    if db_type.lower() == "sql":
        quest = "?, " * len(words_in)
        asset.execute("SELECT word, phonemes FROM dictionary WHERE word IN ({0})".format(quest[:-2]), words_in)
        result = asset.fetchall()
        d = defaultdict(list)
        for k, v in result:
            d[k].append(v)
        return list(d.items())
    if db_type.lower() == "json":
        words = []
        for k, v in asset.items():
            if k in words_in:
                words.append((k, v))
        return words


def get_cmu(tokens_in, db_type="sql"):
    """query the SQL database for the words and return the phonemes in the order of user_in"""
    result = fetch_words(tokens_in, db_type)
    ordered = []
    for word in tokens_in:
        this_word = [[i[1] for i in result if i[0] == word]][0]
        if this_word:
            ordered.append(this_word[0])
        else:
            ordered.append(["__IGNORE__" + word])
    return ordered


def cmu_to_ipa(cmu_list, mark=True, stress_marking='all'):
    """converts the CMU word lists into IPA transcriptions"""
    symbols = {"a": "ə", "ey": "eɪ", "aa": "ɑ", "ae": "æ", "ah": "ə", "ao": "ɔ",
               "aw": "aʊ", "ay": "aɪ", "ch": "ʧ", "dh": "ð", "eh": "ɛ", "er": "ər",
               "hh": "h", "ih": "ɪ", "jh": "ʤ", "ng": "ŋ",  "ow": "oʊ", "oy": "ɔɪ",
               "sh": "ʃ", "th": "θ", "uh": "ʊ", "uw": "u", "zh": "ʒ", "iy": "i", "y": "j"}
    ipa_list = []  # the final list of IPA tokens to be returned
    for word_list in cmu_list:
        ipa_word_list = []  # the word list for each word
        for word in word_list:
            if stress_marking:
                word = stress.find_stress(word, type=stress_marking)
            else:
                if re.sub("\d*", "", word.replace("__IGNORE__", "")) == "":
                    pass  # do not delete token if it's all numbers
                else:
                    word = re.sub("[0-9]", "", word)
            ipa_form = ''
            if word.startswith("__IGNORE__"):
                ipa_form = word.replace("__IGNORE__", "")
                # mark words we couldn't transliterate with an asterisk:

                if mark:
                    if not re.sub("\d*", "", ipa_form) == "":
                        ipa_form += "*"
            else:
                for piece in word.split(" "):
                    marked = False
                    unmarked = piece
                    if piece[0] in ["ˈ", "ˌ"]:
                        marked = True
                        mark = piece[0]
                        unmarked = piece[1:]
                    if unmarked in symbols:
                        if marked:
                            ipa_form += mark + symbols[unmarked]
                        else:
                            ipa_form += symbols[unmarked]

                    else:
                        ipa_form += piece
            swap_list = [["ˈər", "əˈr"], ["ˈie", "iˈe"]]
            for sym in swap_list:
                if not ipa_form.startswith(sym[0]):
                    ipa_form = ipa_form.replace(sym[0], sym[1])
            ipa_word_list.append(ipa_form)
        ipa_list.append(sorted(list(set(ipa_word_list))))
    return ipa_list


def get_top(ipa_list):
    """Returns only the one result for a query. If multiple entries for words are found, only the first is used."""
    return ' '.join([word_list[-1] for word_list in ipa_list])


def get_all(ipa_list):
    """utilizes an algorithm to discover and return all possible combinations of IPA transcriptions"""
    final_size = 1
    for word_list in ipa_list:
        final_size *= len(word_list)
    list_all = ["" for s in range(final_size)]
    for i in range(len(ipa_list)):
        if i == 0:
            swtich_rate = final_size / len(ipa_list[i])
        else:
            swtich_rate /= len(ipa_list[i])
        k = 0
        for j in range(final_size):
            if (j+1) % int(swtich_rate) == 0:
                k += 1
            if k == len(ipa_list[i]):
                k = 0
            list_all[j] = list_all[j] + ipa_list[i][k] + " "
    return sorted([sent[:-1] for sent in list_all])


def ipa_list(words_in, keep_punct=True, stress_marks='both', db_type="sql"):
    """Returns a list of all the discovered IPA transcriptions for each word."""
    if type(words_in) == str:
        words = [preserve_punc(w.lower())[0] for w in words_in.split()]
    else:
        words = [preserve_punc(w.lower())[0] for w in words_in]
    cmu = get_cmu([w[1] for w in words], db_type=db_type)
    ipa = cmu_to_ipa(cmu, stress_marking=stress_marks)
    if keep_punct:
        ipa = _punct_replace_word(words, ipa)
    return ipa


def isin_cmu(word, db_type="sql"):
    """checks if a word is in the CMU dictionary. Doesn't strip punctuation.
    If given more than one word, returns True only if all words are present."""
    if type(word) == str:
        word = [preprocess(w) for w in word.split()]
    results = fetch_words(word, db_type)
    as_set = list(set(t[0] for t in results))
    return len(as_set) == len(set(word))


def convert(text, retrieve_all=False, keep_punct=True, stress_marks='both', mode="sql"):
    """takes either a string or list of English words and converts them to IPA"""
    ipa = ipa_list(
                   words_in=text,
                   keep_punct=keep_punct,
                   stress_marks=stress_marks,
                   db_type=mode)
    if retrieve_all:
        return get_all(ipa)
    return get_top(ipa)


def jonvert(text, retrieve_all=False, keep_punct=True, stress_marks='both'):
    """Forces use of JSON database for fetching phoneme data."""
    return convert(text, retrieve_all, keep_punct, stress_marks, mode="json")
