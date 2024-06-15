import asyncio
import collections
import itertools
import queue
import threading

from typing import AsyncIterable, Callable, Iterable, List, Optional, Union

from ctranslate2._ext import (
    GenerationResult,
    GenerationStepResult,
    Generator,
    ScoringResult,
    TranslationResult,
    Translator,
)


def register_extensions():
    """Registers additional attributes to compiled modules."""
    setattr(Translator, "translate_iterable", translator_translate_iterable)
    setattr(Translator, "score_iterable", translator_score_iterable)
    setattr(Translator, "generate_tokens", translator_generate_tokens)
    setattr(Generator, "generate_iterable", generator_generate_iterable)
    setattr(Generator, "score_iterable", generator_score_iterable)
    setattr(Generator, "generate_tokens", generator_generate_tokens)
    setattr(Generator, "async_generate_tokens", generator_async_generate_tokens)


def translator_translate_iterable(
    translator: Translator,
    source: Iterable[List[str]],
    target_prefix: Optional[Iterable[List[str]]] = None,
    max_batch_size: int = 32,
    batch_type: str = "examples",
    **kwargs,
) -> Iterable[TranslationResult]:
    """Translates an iterable of tokenized examples.

    This method is built on top of :meth:`ctranslate2.Translator.translate_batch`
    to efficiently translate an arbitrarily large stream of data. It enables the
    following optimizations:

    * stream processing (the iterable is not fully materialized in memory)
    * parallel translations (if the translator has multiple workers)
    * asynchronous batch prefetching
    * local sorting by length

    Arguments:
      source: An iterable of tokenized source examples.
      target_prefix: An optional iterable of tokenized target prefixes.
      max_batch_size: The maximum batch size.
      batch_type: Whether :obj:`max_batch_size` is the number of "examples" or "tokens".
      **kwargs: Any translation options accepted by
        :meth:`ctranslate2.Translator.translate_batch`.

    Returns:
      A generator iterator over :class:`ctranslate2.TranslationResult` instances.

    Example:
      This method can be used to efficiently translate text files:

      .. code-block:: python

          # Replace by your own tokenization and detokenization functions.
          tokenize_fn = lambda line: line.strip().split()
          detokenize_fn = lambda tokens: " ".join(tokens)

          with open("input.txt") as input_file:
              source = map(tokenize_fn, input_file)
              results = translator.translate_iterable(source, max_batch_size=64)

              for result in results:
                  tokens = result.hypotheses[0]
                  target = detokenize_fn(tokens)
                  print(target)
    """
    iterables = [source]
    if target_prefix is not None:
        iterables.append(target_prefix)

    yield from _process_iterable(
        translator.translate_batch,
        iterables,
        max_batch_size,
        batch_type,
        **kwargs,
    )


def translator_score_iterable(
    translator: Translator,
    source: Iterable[List[str]],
    target: Iterable[List[str]],
    max_batch_size: int = 64,
    batch_type: str = "examples",
    **kwargs,
) -> Iterable[ScoringResult]:
    """Scores an iterable of tokenized examples.

    This method is built on top of :meth:`ctranslate2.Translator.score_batch`
    to efficiently score an arbitrarily large stream of data. It enables the
    following optimizations:

    * stream processing (the iterable is not fully materialized in memory)
    * parallel scoring (if the translator has multiple workers)
    * asynchronous batch prefetching
    * local sorting by length

    Arguments:
      source: An iterable of tokenized source examples.
      target: An iterable of tokenized target examples.
      max_batch_size: The maximum batch size.
      batch_type: Whether :obj:`max_batch_size` is the number of "examples" or "tokens".
      **kwargs: Any scoring options accepted by
        :meth:`ctranslate2.Translator.score_batch`.

    Returns:
      A generator iterator over :class:`ctranslate2.ScoringResult` instances.
    """
    yield from _process_iterable(
        translator.score_batch,
        [source, target],
        max_batch_size,
        batch_type,
        **kwargs,
    )


def generator_generate_iterable(
    generator: Generator,
    start_tokens: Iterable[List[str]],
    max_batch_size: int = 32,
    batch_type: str = "examples",
    **kwargs,
) -> Iterable[GenerationResult]:
    """Generates from an iterable of tokenized prompts.

    This method is built on top of :meth:`ctranslate2.Generator.generate_batch`
    to efficiently run generation on an arbitrarily large stream of data. It enables
    the following optimizations:

    * stream processing (the iterable is not fully materialized in memory)
    * parallel generations (if the generator has multiple workers)
    * asynchronous batch prefetching
    * local sorting by length

    Arguments:
      start_tokens: An iterable of tokenized prompts.
      max_batch_size: The maximum batch size.
      batch_type: Whether :obj:`max_batch_size` is the number of "examples" or "tokens".
      **kwargs: Any generation options accepted by
        :meth:`ctranslate2.Generator.generate_batch`.

    Returns:
      A generator iterator over :class:`ctranslate2.GenerationResult` instances.
    """
    yield from _process_iterable(
        generator.generate_batch,
        [start_tokens],
        max_batch_size,
        batch_type,
        **kwargs,
    )


def generator_score_iterable(
    generator: Generator,
    tokens: Iterable[List[str]],
    max_batch_size: int = 64,
    batch_type: str = "examples",
    **kwargs,
) -> Iterable[ScoringResult]:
    """Scores an iterable of tokenized examples.

    This method is built on top of :meth:`ctranslate2.Generator.score_batch`
    to efficiently score an arbitrarily large stream of data. It enables
    the following optimizations:

    * stream processing (the iterable is not fully materialized in memory)
    * parallel scoring (if the generator has multiple workers)
    * asynchronous batch prefetching
    * local sorting by length

    Arguments:
      tokens: An iterable of tokenized examples.
      max_batch_size: The maximum batch size.
      batch_type: Whether :obj:`max_batch_size` is the number of "examples" or "tokens".
      **kwargs: Any score options accepted by
        :meth:`ctranslate2.Generator.score_batch`.

    Returns:
      A generator iterator over :class:`ctranslate2.ScoringResult` instances.
    """
    yield from _process_iterable(
        generator.score_batch,
        [tokens],
        max_batch_size,
        batch_type,
        **kwargs,
    )


def translator_generate_tokens(
    translator: Translator,
    source: List[str],
    target_prefix: Optional[List[str]] = None,
    *,
    max_decoding_length: int = 256,
    min_decoding_length: int = 1,
    sampling_topk: int = 1,
    sampling_topp: float = 1,
    sampling_temperature: float = 1,
    return_log_prob: bool = False,
    repetition_penalty: float = 1,
    no_repeat_ngram_size: int = 0,
    disable_unk: bool = False,
    suppress_sequences: Optional[List[List[str]]] = None,
    end_token: Optional[Union[str, List[str], List[int]]] = None,
    max_input_length: int = 1024,
    use_vmap: bool = False,
) -> Iterable[GenerationStepResult]:
    """Yields tokens as they are generated by the model.

    Arguments:
      source: Source tokens.
      target_prefix: Optional target prefix tokens.
      max_decoding_length: Maximum prediction length.
      min_decoding_length: Minimum prediction length.
      sampling_topk: Randomly sample predictions from the top K candidates.
      sampling_topp: Keep the most probable tokens whose cumulative probability exceeds this value.
      sampling_temperature: Sampling temperature to generate more random samples.
      return_log_prob: Include the token log probability in the result.
      repetition_penalty: Penalty applied to the score of previously generated tokens
        (set > 1 to penalize).
      no_repeat_ngram_size: Prevent repetitions of ngrams with this size
        (set 0 to disable).
      disable_unk: Disable the generation of the unknown token.
      suppress_sequences: Disable the generation of some sequences of tokens.
      end_token: Stop the decoding on one of these tokens (defaults to the model EOS token).
      max_input_length: Truncate inputs after this many tokens (set 0 to disable).
      use_vmap: Use the vocabulary mapping file saved in this model

    Returns:
      A generator iterator over :class:`ctranslate2.GenerationStepResult` instances.

    Note:
      This generation method is not compatible with beam search which requires a complete decoding.
    """
    yield from _generate_tokens(
        translator.translate_batch,
        [source],
        [target_prefix] if target_prefix is not None else None,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        disable_unk=disable_unk,
        suppress_sequences=suppress_sequences,
        end_token=end_token,
        max_decoding_length=max_decoding_length,
        min_decoding_length=min_decoding_length,
        sampling_topk=sampling_topk,
        sampling_topp=sampling_topp,
        sampling_temperature=sampling_temperature,
        return_scores=return_log_prob,
        max_input_length=max_input_length,
        use_vmap=use_vmap,
    )


def generator_generate_tokens(
    generator: Generator,
    prompt: Union[List[str], List[List[str]]],
    max_batch_size: int = 0,
    batch_type: str = "examples",
    *,
    max_length: int = 512,
    min_length: int = 0,
    sampling_topk: int = 1,
    sampling_topp: float = 1,
    sampling_temperature: float = 1,
    return_log_prob: bool = False,
    repetition_penalty: float = 1,
    no_repeat_ngram_size: int = 0,
    disable_unk: bool = False,
    suppress_sequences: Optional[List[List[str]]] = None,
    end_token: Optional[Union[str, List[str], List[int]]] = None,
    static_prompt: Optional[List[str]] = None,
    cache_static_prompt: bool = True,
    callback: Callable[[GenerationStepResult], bool] = None,
) -> Iterable[GenerationStepResult]:
    """Yields tokens as they are generated by the model.

    Arguments:
      prompt: Batch of start tokens. If the decoder starts from a
        special start token like <s>, this token should be added to this input.
      max_batch_size: The maximum batch size.
      batch_type: Whether :obj:`max_batch_size` is the number of "examples" or "tokens".
      max_length: Maximum generation length.
      min_length: Minimum generation length.
      sampling_topk: Randomly sample predictions from the top K candidates.
      sampling_topp: Keep the most probable tokens whose cumulative probability exceeds this value.
      sampling_temperature: Sampling temperature to generate more random samples.
      return_log_prob: Include the token log probability in the result.
      repetition_penalty: Penalty applied to the score of previously generated tokens
        (set > 1 to penalize).
      no_repeat_ngram_size: Prevent repetitions of ngrams with this size
        (set 0 to disable).
      disable_unk: Disable the generation of the unknown token.
      suppress_sequences: Disable the generation of some sequences of tokens.
      end_token: Stop the decoding on one these tokens (defaults to the model EOS token).
      static_prompt: If the model expects a static prompt (a.k.a. system prompt)
        it can be set here to simplify the inputs and optionally cache the model
        state for this prompt to accelerate future generations.
      cache_static_prompt: Cache the model state after the static prompt and
        reuse it for future generations using the same static prompt.
      callback: Optional function that is called for each generated token when
        obj:`beam_size` is 1. If the callback function returns ``True``, the
        decoding will stop for this batch index.

    Returns:
      A generator iterator over :class:`ctranslate2.GenerationStepResult` instances.

    Note:
      This generation method is not compatible with beam search which requires a complete decoding.
    """
    if len(prompt) > 0 and isinstance(prompt[0], str):
        prompt = [prompt]
    yield from _generate_tokens(
        generator.generate_batch,
        prompt,
        max_batch_size=max_batch_size,
        batch_type=batch_type,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        disable_unk=disable_unk,
        suppress_sequences=suppress_sequences,
        end_token=end_token,
        max_length=max_length,
        min_length=min_length,
        sampling_topk=sampling_topk,
        sampling_topp=sampling_topp,
        sampling_temperature=sampling_temperature,
        return_scores=return_log_prob,
        static_prompt=static_prompt,
        cache_static_prompt=cache_static_prompt,
        include_prompt_in_result=False,
        callback=callback,
    )


async def generator_async_generate_tokens(
    generator: Generator,
    prompt: Union[List[str], List[List[str]]],
    max_batch_size: int = 0,
    batch_type: str = "examples",
    *,
    max_length: int = 512,
    min_length: int = 0,
    sampling_topk: int = 1,
    sampling_topp: float = 1,
    sampling_temperature: float = 1,
    return_log_prob: bool = False,
    repetition_penalty: float = 1,
    no_repeat_ngram_size: int = 0,
    disable_unk: bool = False,
    suppress_sequences: Optional[List[List[str]]] = None,
    end_token: Optional[Union[str, List[str], List[int]]] = None,
    static_prompt: Optional[List[str]] = None,
    cache_static_prompt: bool = True,
    callback: Callable[[GenerationStepResult], bool] = None,
) -> AsyncIterable[GenerationStepResult]:
    """Yields tokens asynchronously as they are generated by the model.

    Arguments:
      prompt: Batch of start tokens. If the decoder starts from a
        special start token like <s>, this token should be added to this input.
      max_batch_size: The maximum batch size.
      batch_type: Whether :obj:`max_batch_size` is the number of "examples" or "tokens".
      max_length: Maximum generation length.
      min_length: Minimum generation length.
      sampling_topk: Randomly sample predictions from the top K candidates.
      sampling_topp: Keep the most probable tokens whose cumulative probability exceeds this value.
      sampling_temperature: Sampling temperature to generate more random samples.
      return_log_prob: Include the token log probability in the result.
      repetition_penalty: Penalty applied to the score of previously generated tokens
        (set > 1 to penalize).
      no_repeat_ngram_size: Prevent repetitions of ngrams with this size
        (set 0 to disable).
      disable_unk: Disable the generation of the unknown token.
      suppress_sequences: Disable the generation of some sequences of tokens.
      end_token: Stop the decoding on one of these tokens (defaults to the model EOS token).
      static_prompt: If the model expects a static prompt (a.k.a. system prompt)
        it can be set here to simplify the inputs and optionally cache the model
        state for this prompt to accelerate future generations.
      cache_static_prompt: Cache the model state after the static prompt and
        reuse it for future generations using the same static prompt.
      callback: Optional function that is called for each generated token when
        obj:`beam_size` is 1. If the callback function returns ``True``, the
        decoding will stop for this batch index.

    Returns:
      An async generator iterator over :class:`ctranslate2.GenerationStepResult` instances.

    Note:
      This generation method is not compatible with beam search which requires a complete decoding.
    """
    if len(prompt) > 0 and isinstance(prompt[0], str):
        prompt = [prompt]
    async for step_result in AsyncGenerator(
        generator.generate_batch,
        prompt,
        max_batch_size=max_batch_size,
        batch_type=batch_type,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        disable_unk=disable_unk,
        suppress_sequences=suppress_sequences,
        end_token=end_token,
        max_length=max_length,
        min_length=min_length,
        sampling_topk=sampling_topk,
        sampling_topp=sampling_topp,
        sampling_temperature=sampling_temperature,
        return_scores=return_log_prob,
        static_prompt=static_prompt,
        cache_static_prompt=cache_static_prompt,
        include_prompt_in_result=False,
        callback=callback,
    ):
        yield step_result


class AsyncGenerator:
    def __init__(self, process_func, *args, **kwargs):
        self.queue = asyncio.Queue()
        self.shutdown_event = threading.Event()
        self.iterator_task = None
        self.process_func = process_func
        self.args = args
        self.kwargs = kwargs

    async def producer(self):
        # Data generation logic here
        for step_result in _generate_tokens(
            self.process_func, *self.args, **self.kwargs
        ):
            await self.queue.put(step_result)
            await asyncio.sleep(0.0001)
            # asyc sleep otherwise this doesn't yield any result
            if self.shutdown_event.is_set():
                break
        await self.queue.put(None)

    def __aiter__(self):
        self.iterator_task = asyncio.create_task(self.producer())
        return self

    async def __anext__(self):
        if self.shutdown_event.is_set():
            raise StopAsyncIteration

        try:
            item = await self.queue.get()
            if item is None:
                self.shutdown_event.set()
                raise StopAsyncIteration
            return item
        except asyncio.CancelledError:
            self.shutdown_event.set()
            raise StopAsyncIteration


def _generate_tokens(process_func, *args, **kwargs):
    step_results = queue.Queue()
    generator_closed = threading.Event()

    user_callback = kwargs.get("callback", None)
    if user_callback is None:
        user_callback = lambda step_result: False

    def _callback(step_result):
        user_callback_result = user_callback(step_result)
        step_results.put(step_result)

        return generator_closed.is_set() or user_callback_result

    kwargs.update(
        {
            "asynchronous": True,
            "beam_size": 1,
            "callback": _callback,
        }
    )

    async_results = process_func(*args, **kwargs)

    def _catch_exception():
        try:
            for result in async_results:
                result.result()
        except Exception as e:
            step_results.put(e)
        step_results.put(None)

    thread = threading.Thread(target=_catch_exception, daemon=True)
    thread.start()

    while True:
        step_result = step_results.get()

        if step_result is None:
            break

        if isinstance(step_result, Exception):
            raise step_result

        try:
            yield step_result
        except GeneratorExit:
            generator_closed.set()
            break

    # Wait for the job to terminate before exiting.
    thread.join()


def _process_iterable(process_func, iterables, max_batch_size, batch_type, **kwargs):
    if max_batch_size < 1:
        raise ValueError("max_batch_size must be >= 1")

    if len(iterables) == 1:
        iterable = iterables[0]
    else:
        iterable = itertools.zip_longest(*iterables)

    kwargs.update(
        {
            "max_batch_size": max_batch_size,
            "batch_type": batch_type,
            "asynchronous": True,
        }
    )

    read_batch_size = max_batch_size * 16 if max_batch_size > 1 else max_batch_size
    queue = collections.deque()

    for streams in _batch_iterator(iterable, read_batch_size, batch_type):
        queue.extend(process_func(*streams, **kwargs))

        while queue and queue[0].done():
            yield queue.popleft().result()

    while queue:
        yield queue.popleft().result()


def _batch_iterator(iterable, batch_size, batch_type):
    streams = None
    cur_batch_size = 0

    for example in iterable:
        if not isinstance(example, tuple):
            example = (example,)

        if streams is None:
            streams = tuple([] for _ in example)
        for batch, element in zip(streams, example):
            if element is None and len(streams) > 1:
                raise ValueError("Input iterables do not have the same length")
            batch.append(element)

        if batch_type == "examples":
            cur_batch_size += 1
        elif batch_type == "tokens":
            cur_batch_size += len(example[0])
        else:
            raise ValueError("Invalid batch type %s" % batch_type)

        if cur_batch_size >= batch_size:
            yield streams
            streams = None
            cur_batch_size = 0

    if streams is not None:
        yield streams
