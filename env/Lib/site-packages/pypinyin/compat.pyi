# -*- coding: utf-8 -*-
from typing import Any, List, Text, ByteString

SUPPORT_UCS4 = ...  # type: bool

PY2 = ...  # type: bool

subversion = ...  # type: List[Text]

text_type = ...  # type: Text
bytes_type = ...  # type: ByteString


def callable_check(obj: Any) -> bool: ...
