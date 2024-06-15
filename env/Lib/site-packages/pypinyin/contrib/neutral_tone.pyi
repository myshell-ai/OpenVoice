# -*- coding: utf-8 -*-
from typing import Any
from typing import Optional
from typing import Text
from typing import Tuple

from pypinyin.constants import Style

TStyle = Style


class NeutralToneWith5Mixin(object):
    NUMBER_TONE = ...  # type: Tuple[TStyle]
    NUMBER_AT_END = ...  # type: Tuple[TStyle]

    def post_convert_style(self, han: Text, orig_pinyin: Text,
                           converted_pinyin: Text, style: TStyle,
                           strict: bool, **kwargs: Any) -> Optional[Text]: ...
