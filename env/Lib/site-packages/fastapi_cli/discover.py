import importlib
import sys
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Union

from rich import print
from rich.padding import Padding
from rich.panel import Panel
from rich.syntax import Syntax
from rich.tree import Tree

from fastapi_cli.exceptions import FastAPICLIException

logger = getLogger(__name__)

try:
    from fastapi import FastAPI
except ImportError:  # pragma: no cover
    FastAPI = None  # type: ignore[misc, assignment]


def get_default_path() -> Path:
    path = Path("main.py")
    if path.is_file():
        return path
    path = Path("app.py")
    if path.is_file():
        return path
    path = Path("api.py")
    if path.is_file():
        return path
    path = Path("app/main.py")
    if path.is_file():
        return path
    path = Path("app/app.py")
    if path.is_file():
        return path
    path = Path("app/api.py")
    if path.is_file():
        return path
    raise FastAPICLIException(
        "Could not find a default file to run, please provide an explicit path"
    )


@dataclass
class ModuleData:
    module_import_str: str
    extra_sys_path: Path


def get_module_data_from_path(path: Path) -> ModuleData:
    logger.info(
        "Searching for package file structure from directories with [blue]__init__.py[/blue] files"
    )
    use_path = path.resolve()
    module_path = use_path
    if use_path.is_file() and use_path.stem == "__init__":
        module_path = use_path.parent
    module_paths = [module_path]
    extra_sys_path = module_path.parent
    for parent in module_path.parents:
        init_path = parent / "__init__.py"
        if init_path.is_file():
            module_paths.insert(0, parent)
            extra_sys_path = parent.parent
        else:
            break
    logger.info(f"Importing from {extra_sys_path.resolve()}")
    root = module_paths[0]
    name = f"🐍 {root.name}" if root.is_file() else f"📁 {root.name}"
    root_tree = Tree(name)
    if root.is_dir():
        root_tree.add("[dim]🐍 __init__.py[/dim]")
    tree = root_tree
    for sub_path in module_paths[1:]:
        sub_name = (
            f"🐍 {sub_path.name}" if sub_path.is_file() else f"📁 {sub_path.name}"
        )
        tree = tree.add(sub_name)
        if sub_path.is_dir():
            tree.add("[dim]🐍 __init__.py[/dim]")
    title = "[b green]Python module file[/b green]"
    if len(module_paths) > 1 or module_path.is_dir():
        title = "[b green]Python package file structure[/b green]"
    panel = Padding(
        Panel(
            root_tree,
            title=title,
            expand=False,
            padding=(1, 2),
        ),
        1,
    )
    print(panel)
    module_str = ".".join(p.stem for p in module_paths)
    logger.info(f"Importing module [green]{module_str}[/green]")
    return ModuleData(
        module_import_str=module_str, extra_sys_path=extra_sys_path.resolve()
    )


def get_app_name(*, mod_data: ModuleData, app_name: Union[str, None] = None) -> str:
    try:
        mod = importlib.import_module(mod_data.module_import_str)
    except (ImportError, ValueError) as e:
        logger.error(f"Import error: {e}")
        logger.warning(
            "Ensure all the package directories have an [blue]__init__.py[/blue] file"
        )
        raise
    if not FastAPI:  # type: ignore[truthy-function]
        raise FastAPICLIException(
            "Could not import FastAPI, try running 'pip install fastapi'"
        ) from None
    object_names = dir(mod)
    object_names_set = set(object_names)
    if app_name:
        if app_name not in object_names_set:
            raise FastAPICLIException(
                f"Could not find app name {app_name} in {mod_data.module_import_str}"
            )
        app = getattr(mod, app_name)
        if not isinstance(app, FastAPI):
            raise FastAPICLIException(
                f"The app name {app_name} in {mod_data.module_import_str} doesn't seem to be a FastAPI app"
            )
        return app_name
    for preferred_name in ["app", "api"]:
        if preferred_name in object_names_set:
            obj = getattr(mod, preferred_name)
            if isinstance(obj, FastAPI):
                return preferred_name
    for name in object_names:
        obj = getattr(mod, name)
        if isinstance(obj, FastAPI):
            return name
    raise FastAPICLIException("Could not find FastAPI app in module, try using --app")


def get_import_string(
    *, path: Union[Path, None] = None, app_name: Union[str, None] = None
) -> str:
    if not path:
        path = get_default_path()
    logger.info(f"Using path [blue]{path}[/blue]")
    logger.info(f"Resolved absolute path {path.resolve()}")
    if not path.exists():
        raise FastAPICLIException(f"Path does not exist {path}")
    mod_data = get_module_data_from_path(path)
    sys.path.insert(0, str(mod_data.extra_sys_path))
    use_app_name = get_app_name(mod_data=mod_data, app_name=app_name)
    import_example = Syntax(
        f"from {mod_data.module_import_str} import {use_app_name}", "python"
    )
    import_panel = Padding(
        Panel(
            import_example,
            title="[b green]Importable FastAPI app[/b green]",
            expand=False,
            padding=(1, 2),
        ),
        1,
    )
    logger.info("Found importable FastAPI app")
    print(import_panel)
    import_string = f"{mod_data.module_import_str}:{use_app_name}"
    logger.info(f"Using import string [b green]{import_string}[/b green]")
    return import_string
