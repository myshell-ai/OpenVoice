from typing import Any, Text


class ToneConverter(object):
    def to_tone(self, pinyin: Text, **kwargs: Any) -> Text: ...

    def to_tone2(self, pinyin: Text, **kwargs: Any) -> Text: ...

    def to_tone3(self, pinyin: Text, **kwargs: Any) -> Text: ...


converter = ...  # type: ToneConverter
