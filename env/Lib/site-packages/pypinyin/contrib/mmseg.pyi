from typing import Iterator
from typing import Text


class Seg(object):
    """最大正向匹配分词

    :type prefix_set: PrefixSet
    """
    def __init__(self, prefix_set: PrefixSet) -> None: ...

    def cut(self, text: Text) -> Iterator[Text]: ...

    def train(self, words: Iterator[Text]) -> None: ...


class PrefixSet(object):
    def __init__(self) -> None: ...

    def train(self, word_s: Iterator[Text]) -> None: ...

    def __contains__(self, key: Text) -> bool: ...


p_set = ...  # type: PrefixSet
seg = ...  # type: Seg


def retrain(seg_instance: Seg) -> None: ...
