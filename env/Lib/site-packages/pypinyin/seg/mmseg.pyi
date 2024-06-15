from typing import Iterator
from typing import Text
from typing import Set


class Seg(object):
    """最大正向匹配分词

    :type prefix_set: PrefixSet
    :type no_non_phrases: bool
    """
    def __init__(self, prefix_set: PrefixSet, no_non_phrases: bool) -> None:
        self._no_non_phrases = ...  # type: bool
        self._prefix_set = ...  # type: PrefixSet
        ...

    def cut(self, text: Text) -> Iterator[Text]: ...

    def train(self, words: Iterator[Text]) -> None: ...


class PrefixSet(object):
    def __init__(self) -> None:
        self._set = ...  # type: Set[Text]
        ...

    def train(self, word_s: Iterator[Text]) -> None: ...

    def __contains__(self, key: Text) -> bool: ...


p_set = ...  # type: PrefixSet
seg = ...  # type: Seg


def retrain(seg_instance: Seg) -> None: ...
