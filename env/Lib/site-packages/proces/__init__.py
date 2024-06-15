from .preprocess import get_all_pipelines
from .preprocess import filter_unusual_characters
from .preprocess import handle_blank_character
from .preprocess import uppercase_to_lowercase
from .preprocess import traditional_to_simplified
from .preprocess import full_angle_to_half_angle
from .preprocess import handle_substitute
from .preprocess import preprocess
from .masking import mask_phone
from .masking import mask_address

filter_ = filter_unusual_characters

__version__ = "0.1.7"

__all__ = [
    "get_all_pipelines",
    "filter_unusual_characters", "filter_",
    "handle_blank_character",
    "uppercase_to_lowercase",
    "traditional_to_simplified",
    "full_angle_to_half_angle",
    "handle_substitute",
    "preprocess",
    "mask_phone",
    "mask_address"
]
