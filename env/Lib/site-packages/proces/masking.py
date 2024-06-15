from re import sub

from proces.util.data import get_city_pattern

city_ptn = get_city_pattern()


def mask_phone(text: str) -> str:
    """手机号脱敏

    Attributes:
        text: input text
    """
    text = sub(r"(\+?86|[(（]86[)）])(1\d{2})(\d{8})", r"\1\2********", text)
    text = sub(r"(?<![0-9])(1\d{2})(\d{8})(?![0-9])", r"\1********", text)
    return text


def mask_address(text: str) -> str:
    """地址脱敏

    Attributes:
        text: input text
    """
    text_res = city_ptn.search(text)
    if text_res:
        text = sub(fr"({text_res.group()})\w+", r"\1***", text)
    return text
