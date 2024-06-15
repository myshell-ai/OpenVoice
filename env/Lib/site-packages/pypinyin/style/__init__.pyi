# -*- coding: utf-8 -*-
from typing import Any, Optional, Callable, Dict, Text, Union

from pypinyin.constants import Style

TStyle = Style
TRegisterFunc = Optional[Callable[[Text, Dict[Any, Any]], Text]]
TWrapperFunc = Optional[Callable[[Text, Dict[Any, Any]], Text]]

_registry = {}  # type: Dict[Union[TStyle, int, str, Any], TRegisterFunc]


def convert(pinyin: Text, style: TStyle, strict: bool,
            default: Optional[Text] = ..., **kwargs: Any) -> Text: ...


def register(style: Union[TStyle, int, str, Any],
             func: TRegisterFunc = ...) -> TWrapperFunc: ...

def auto_discover() -> None: ...
