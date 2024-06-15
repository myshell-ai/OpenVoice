from typing import Any, Text


class OthersConverter(object):
    def to_normal(self, pinyin: Text, **kwargs: Any) -> Text: ...

    def to_first_letter(self, pinyin: Text, **kwargs: Any) -> Text: ...


converter = ...  # type: OthersConverter
