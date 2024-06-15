# coding=utf-8
# Copyright 2023-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import TYPE_CHECKING, List, TypedDict


if TYPE_CHECKING:
    from PIL import Image


class ClassificationOutput(TypedDict):
    """Dictionary containing the output of a [`~InferenceClient.audio_classification`] and  [`~InferenceClient.image_classification`] task.

    Args:
        label (`str`):
            The label predicted by the model.
        score (`float`):
            The score of the label predicted by the model.
    """

    label: str
    score: float


class ConversationalOutputConversation(TypedDict):
    """Dictionary containing the "conversation" part of a [`~InferenceClient.conversational`] task.

    Args:
        generated_responses (`List[str]`):
            A list of the responses from the model.
        past_user_inputs (`List[str]`):
            A list of the inputs from the user. Must be the same length as `generated_responses`.
    """

    generated_responses: List[str]
    past_user_inputs: List[str]


class ConversationalOutput(TypedDict):
    """Dictionary containing the output of a  [`~InferenceClient.conversational`] task.

    Args:
        generated_text (`str`):
            The last response from the model.
        conversation (`ConversationalOutputConversation`):
            The past conversation.
        warnings (`List[str]`):
            A list of warnings associated with the process.
    """

    conversation: ConversationalOutputConversation
    generated_text: str
    warnings: List[str]


class FillMaskOutput(TypedDict):
    """Dictionary containing information about a [`~InferenceClient.fill_mask`] task.

    Args:
        score (`float`):
            The probability of the token.
        token (`int`):
            The id of the token.
        token_str (`str`):
            The string representation of the token.
        sequence (`str`):
            The actual sequence of tokens that ran against the model (may contain special tokens).
    """

    score: float
    token: int
    token_str: str
    sequence: str


class ImageSegmentationOutput(TypedDict):
    """Dictionary containing information about a [`~InferenceClient.image_segmentation`] task. In practice, image segmentation returns a
    list of `ImageSegmentationOutput` with 1 item per mask.

    Args:
        label (`str`):
            The label corresponding to the mask.
        mask (`Image`):
            An Image object representing the mask predicted by the model.
        score (`float`):
            The score associated with the label for this mask.
    """

    label: str
    mask: "Image"
    score: float


class ObjectDetectionOutput(TypedDict):
    """Dictionary containing information about a [`~InferenceClient.object_detection`] task.

    Args:
        label (`str`):
            The label corresponding to the detected object.
        box (`dict`):
            A dict response of bounding box coordinates of
            the detected object: xmin, ymin, xmax, ymax
        score (`float`):
            The score corresponding to the detected object.
    """

    label: str
    box: dict
    score: float


class QuestionAnsweringOutput(TypedDict):
    """Dictionary containing information about a [`~InferenceClient.question_answering`] task.

    Args:
        score (`float`):
            A float that represents how likely that the answer is correct.
        start (`int`):
            The index (string wise) of the start of the answer within context.
        end (`int`):
            The index (string wise) of the end of the answer within context.
        answer (`str`):
            A string that is the answer within the text.
    """

    score: float
    start: int
    end: int
    answer: str


class TableQuestionAnsweringOutput(TypedDict):
    """Dictionary containing information about a [`~InferenceClient.table_question_answering`] task.

    Args:
        answer (`str`):
            The plaintext answer.
        coordinates (`List[List[int]]`):
            A list of coordinates of the cells referenced in the answer.
        cells (`List[int]`):
            A list of coordinates of the cells contents.
        aggregator (`str`):
            The aggregator used to get the answer.
    """

    answer: str
    coordinates: List[List[int]]
    cells: List[List[int]]
    aggregator: str


class TokenClassificationOutput(TypedDict):
    """Dictionary containing the output of a [`~InferenceClient.token_classification`] task.

    Args:
        entity_group (`str`):
            The type for the entity being recognized (model specific).
        score (`float`):
            The score of the label predicted by the model.
        word (`str`):
            The string that was captured.
        start (`int`):
            The offset stringwise where the answer is located. Useful to disambiguate if word occurs multiple times.
        end (`int`):
            The offset stringwise where the answer is located. Useful to disambiguate if word occurs multiple times.
    """

    entity_group: str
    score: float
    word: str
    start: int
    end: int
