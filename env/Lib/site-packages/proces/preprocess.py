from re import sub
from typing import Union, Optional

from .conf import T2S_DICT

ALL_PIPELINES = [
    "filter_unusual_characters",
    "handle_blank_character",
    "uppercase_to_lowercase",
    "traditional_to_simplified",
    "full_angle_to_half_angle",
    "handle_substitute"
]


def get_all_pipelines() -> list:
    """获取所有的预处理管道"""
    return ALL_PIPELINES


def filter_unusual_characters(text: str) -> str:
    """过滤所有非常见字符，保留中文、英文、常见标点、空白字符

    Attributes:
        text: input text
    """
    chinese = r"\u4E00-\u9FA5"
    punctuation = r"!\"#$%&\'()*+,\-./:;<=>?@\\\[\]^_`{|}~¥·—‘’“”…、。〈〉《》「」『』【】！（），：；？｜～"

    return sub(fr"[^\w\s{chinese}{punctuation}]+", "", text)


def handle_blank_character(text: str, repl: Optional[str] = "") -> str:
    """处理空白字符，默认替换成空字符

    Attributes:
        text: input text
        repl: replace text
    """
    return sub(r"\s+", repl, text)


def uppercase_to_lowercase(text: str) -> str:
    """大写转小写

    Attributes:
        text: input text
    """
    return text.lower()


def traditional_to_simplified(text: str) -> str:
    """繁体转简体

    Attributes:
        text: input text

    convert data from mediawiki.
    """
    return "".join([T2S_DICT[t] if t in T2S_DICT.keys() else t for t in text])


def full_angle_to_half_angle(text: str) -> str:
    """全角转半角

    Attributes:
        text: input text
    """
    result = ""
    for uchar in text:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif 65281 <= inside_code <= 65374:  # 全角字符（除空格）根据关系转化
            inside_code -= 65248

        result += chr(inside_code)
    return result


def handle_substitute(text: str, ptn: str, repl: str) -> str:
    """替换一些字符

    Attributes:
        text: input text
        ptn: re pattern
        repl: replace text
    """
    return sub(ptn, repl, text)


def preprocess(data: Union[str, list], pipelines: Optional[list] = None, params: Optional[dict] = None) \
        -> Union[str, list]:
    """文本预处理

    Attributes:
        data: input data
        pipelines: default is
            ["handle_blank_character",
            "uppercase_to_lowercase",
            "traditional_to_simplified",
            "full_angle_to_half_angle"]
        params: function parameters
    """
    default_pipelines = [
        "handle_blank_character",
        "uppercase_to_lowercase",
        "traditional_to_simplified",
        "full_angle_to_half_angle"
    ]
    if pipelines is None:
        pipelines = default_pipelines

    if type(data) == str:
        data_list = [data]
    else:
        data_list = data

    results = []
    for text in data_list:
        for func in pipelines:
            if func in ALL_PIPELINES:
                if params is None:
                    text = globals()[func](text)
                else:
                    if func in params.keys():
                        text = globals()[func](text, *params[func])
            else:
                raise ValueError(f"pipeline: {func} not support!")
        results.append(text)

    if type(data) == str:
        return results[0]
    else:
        return results
