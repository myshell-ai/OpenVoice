from typing import Any, Text


class FinalsConverter(object):
    def to_finals(self, pinyin: Text, **kwargs: Any) -> Text: ...

    def to_finals_tone(self, pinyin: Text, **kwargs: Any) -> Text: ...

    def to_finals_tone2(self, pinyin: Text, **kwargs: Any) -> Text: ...

    def to_finals_tone3(self, pinyin: Text, **kwargs: Any) -> Text: ...


converter = ...  # type: FinalsConverter
