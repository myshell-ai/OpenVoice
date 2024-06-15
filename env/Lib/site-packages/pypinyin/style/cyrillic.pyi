from typing import Any, Dict, Tuple, Text

# 俄语转换表
CYRILLIC_REPLACE = ...  # type: Tuple[Tuple[Any]]
CYRILLIC_TABLE = ...  # type: Dict[Text, Text]


class CyrillicfoConverter(object):
    def to_cyrillic(self, pinyin: Text, **kwargs: Any) -> Text: ...

    def to_cyrillic_first(self, pinyin: Text, **kwargs: Any) -> Text: ...

    def _pre_convert(self, pinyin: Text) -> Text: ...


converter = ...  # type: CyrillicfoConverter
