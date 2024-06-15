import json
from contextlib import suppress
from pathlib import PurePath
from typing import Any, Callable, Dict, List, Optional, Tuple

from .registry import _import_class, get_filesystem_class
from .spec import AbstractFileSystem


class FilesystemJSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, AbstractFileSystem):
            return o.to_dict()
        if isinstance(o, PurePath):
            cls = type(o)
            return {"cls": f"{cls.__module__}.{cls.__name__}", "str": str(o)}

        return super().default(o)


class FilesystemJSONDecoder(json.JSONDecoder):
    def __init__(
        self,
        *,
        object_hook: Optional[Callable[[Dict[str, Any]], Any]] = None,
        parse_float: Optional[Callable[[str], Any]] = None,
        parse_int: Optional[Callable[[str], Any]] = None,
        parse_constant: Optional[Callable[[str], Any]] = None,
        strict: bool = True,
        object_pairs_hook: Optional[Callable[[List[Tuple[str, Any]]], Any]] = None,
    ) -> None:
        self.original_object_hook = object_hook

        super().__init__(
            object_hook=self.custom_object_hook,
            parse_float=parse_float,
            parse_int=parse_int,
            parse_constant=parse_constant,
            strict=strict,
            object_pairs_hook=object_pairs_hook,
        )

    @classmethod
    def try_resolve_path_cls(cls, dct: Dict[str, Any]):
        with suppress(Exception):
            fqp = dct["cls"]

            path_cls = _import_class(fqp)

            if issubclass(path_cls, PurePath):
                return path_cls

        return None

    @classmethod
    def try_resolve_fs_cls(cls, dct: Dict[str, Any]):
        with suppress(Exception):
            if "cls" in dct:
                try:
                    fs_cls = _import_class(dct["cls"])
                    if issubclass(fs_cls, AbstractFileSystem):
                        return fs_cls
                except Exception:
                    if "protocol" in dct:  # Fallback if cls cannot be imported
                        return get_filesystem_class(dct["protocol"])

                    raise

        return None

    def custom_object_hook(self, dct: Dict[str, Any]):
        if "cls" in dct:
            if (obj_cls := self.try_resolve_fs_cls(dct)) is not None:
                return AbstractFileSystem.from_dict(dct)
            if (obj_cls := self.try_resolve_path_cls(dct)) is not None:
                return obj_cls(dct["str"])

        if self.original_object_hook is not None:
            return self.original_object_hook(dct)

        return dct
