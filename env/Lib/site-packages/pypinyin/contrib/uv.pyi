# -*- coding: utf-8 -*-

from typing import Any
from typing import Optional
from typing import Text

from pypinyin.constants import Style

TStyle = Style


class V2UMixin(object):

    def post_convert_style(self, han: Text, orig_pinyin: Text,
                           converted_pinyin: Text, style: TStyle,
                           strict: bool, **kwargs: Any) -> Optional[Text]: ...
