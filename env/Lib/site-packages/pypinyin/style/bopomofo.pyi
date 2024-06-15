from typing import Any, Dict, Tuple, Text

# 注音转换表
BOPOMOFO_REPLACE = ...  # type: Tuple[Tuple[Any]]
BOPOMOFO_TABLE = ...  # type: Dict[Text, Text]


class BopomofoConverter(object):
    def to_bopomofo(self, pinyin: Text, **kwargs: Any) -> Text: ...

    def to_bopomofo_first(self, pinyin: Text, **kwargs: Any) -> Text: ...

    def _pre_convert(self, pinyin: Text) -> Text: ...

converter = ...  # type: BopomofoConverter
