import os
import re
import json
import eng_to_ipa.syllables as syllables
import logging


def create_phones_json():
    """Creates the phones.json file in the resources directory from the phones.txt source file from CMU"""
    phones_dict = {}
    with open(os.path.join(os.path.abspath(os.path.dirname(__file__)),
                           'resources','CMU_source_files','cmudict-0.7b.phones.txt'), encoding="UTF-8") as phones_txt:
        # source link: http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b.phones
        for line in phones_txt.readlines():
            phones_dict[line.split("	")[0].lower()] = line.split("	")[1].replace("\n", "")

    with open(os.path.join(os.path.abspath(os.path.dirname(__file__)),
                           'resources','phones.json'), "w") as phones_json:
        json.dump(phones_dict, phones_json)


def stress_type(stress):
    """Determine the kind of stress that should be evaluated"""
    stress = stress.lower()
    default = {"1": "ˈ", "2": "ˌ"}
    if stress == "primary":
        return {"1": "ˈ"}
    elif stress == "secondary":
        return {"2": "ˌ"}
    elif stress == "both" or stress == "all":
        return default
    else:
        logging.warning("WARNING: stress type parameter " + stress + " not recognized.")
        # Use default stress
        return default

with open(os.path.join(os.path.abspath(os.path.dirname(__file__)),
                   'resources','phones.json'), "r") as phones_json:
    phones = json.load(phones_json)


def find_stress(word, type="all"):
    """Convert stress marking numbers from CMU into actual stress markings
    :param word: the CMU word string to be evaluated for stress markings
    :param type: type of stress to be evaluated (primary, secondary, or both)"""

    syll_count = syllables.cmu_syllable_count(word)

    if (not word.startswith("__IGNORE__")) and syll_count > 1:
        symbols = word.split(' ')
        stress_map = stress_type(type)
        new_word = []
        clusters = ["sp", "st", "sk", "fr", "fl"]
        stop_set = ["nasal", "fricative", "vowel"]  # stop searching for where stress starts if these are encountered
        # for each CMU symbol
        for c in symbols:
            # if the last character is a 1 or 2 (that means it has stress, and we want to evaluate it)
            if c[-1] in stress_map.keys():
                # if the new_word list is empty
                if not new_word:
                    # append to new_word the CMU symbol, replacing numbers with stress marks
                    new_word.append(re.sub("\d", "", stress_map[re.findall("\d", c)[0]] + c))
                else:
                    stress_mark = stress_map[c[-1]]
                    placed = False
                    hiatus = False
                    new_word = new_word[::-1]  # flip the word and backtrack through symbols
                    for i, sym in enumerate(new_word):
                        sym = re.sub("[0-9ˈˌ]", "", sym)
                        prev_sym = re.sub("[0-9ˈˌ]", "", new_word[i-1])
                        prev_phone = phones[re.sub("[0-9ˈˌ]", "", new_word[i-1])]
                        if phones[sym] in stop_set or (i > 0 and prev_phone == "stop") or sym in ["er", "w", "j"]:
                            if sym + prev_sym in clusters:
                                new_word[i] = stress_mark + new_word[i]
                            elif not prev_phone == "vowel" and i > 0:
                                new_word[i-1] = stress_mark + new_word[i-1]
                            else:
                                if phones[sym] == "vowel":
                                    hiatus = True
                                    new_word = [stress_mark + re.sub("[0-9ˈˌ]", "", c)] + new_word
                                else:
                                    new_word[i] = stress_mark + new_word[i]
                            placed = True
                            break
                    if not placed:
                        if new_word:
                            new_word[len(new_word) - 1] = stress_mark + new_word[len(new_word) - 1]
                    new_word = new_word[::-1]
                    if not hiatus:
                        new_word.append(re.sub("\d", "", c))
                        hiatus = False
            else:
                if c.startswith("__IGNORE__"):
                    new_word.append(c)
                else:
                    new_word.append(re.sub("\d", "", c))

        return ' '.join(new_word)
    else:
        if word.startswith("__IGNORE__"):
            return word
        else:
            return re.sub("[0-9]", "", word)


if __name__ == "__main__":

    # create phones dictionary from source if not found in the resources directory
    if not os.path.isfile(os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            'resources','phones.json')):
        create_phones_json()
