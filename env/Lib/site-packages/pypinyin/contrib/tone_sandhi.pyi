from typing import Any
from typing import List
from typing import Union
from typing import Callable
from typing import Optional
from typing import Text

from pypinyin.constants import Style


TStyle = Style
TErrors = Union[Callable[[Text], Text], Text]
TPinyinResult = List[List[Text]]
TErrorResult = Union[Text, List[Text], None]
TNoPinyinResult = Union[TPinyinResult, List[Text], Text, None]

class ToneSandhiMixin(object):

    def post_pinyin(self, han: Text, heteronym: bool,
                    pinyin: TPinyinResult,
                    **kwargs: Any) -> Union[TPinyinResult, None]: ...
