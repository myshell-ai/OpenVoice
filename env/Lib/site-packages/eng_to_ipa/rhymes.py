# Simple rhyming support. Call get_rhymes() on a word to find rhymes from the CMU dictionary.
from eng_to_ipa.transcribe import mode_type, get_cmu, preprocess


def remove_onset(word_in):
    phone_list = get_cmu([word_in])[0][0].split(" ")
    for i, phoneme in enumerate(phone_list):
        if "1" in phoneme:
            return ' '.join(phone_list[i:])


def get_rhymes(word, mode="sql"):
    if len(word.split()) > 1:
        return [get_rhymes(w) for w in word.split()]
    phones = remove_onset(preprocess(word))
    phones_full = get_cmu([preprocess(word)])[0][0]
    if mode == "sql":
        c = mode_type(mode)
        c.execute("SELECT word, phonemes FROM dictionary WHERE phonemes " 
                  "LIKE \"%{0}\" AND NOT word=\"{1}\" ".format(phones, word) +
                  "AND NOT phonemes=\"{0}\"".format(phones_full))
        # also don't return results that are the same but spelled differently
        return sorted(list(set([r[0] for r in c.fetchall()])))
    elif mode == "json":
        r_list = []
        for key, val in mode_type(mode).items():
            for v in val:
                if v.endswith(phones) and word != key and v != phones_full:
                    r_list.append(key)
        return sorted(set(r_list))


def jhymes(word):
    """Get rhymes with forced JSON mode."""
    return get_rhymes(word, mode="json")


if __name__ == "__main__":
    test = "testing"
    rhymes = get_rhymes(test)
    for rhyme in rhymes:
        print(rhyme)
