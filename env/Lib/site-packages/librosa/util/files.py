#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility functions for dealing with files"""

import os
import glob
import json
from pathlib import Path

from pkg_resources import resource_filename
import pooch

from .exceptions import ParameterError
from .decorators import deprecate_positional_args


__all__ = [
    "find_files",
    "example",
    "ex",
    "list_examples",
    "example_info",
]


# Instantiate the pooch
__data_path = os.environ.get("LIBROSA_DATA_DIR", pooch.os_cache("librosa"))
__GOODBOY = pooch.create(
    __data_path, base_url="https://librosa.org/data/audio/", registry=None
)

__GOODBOY.load_registry(
    resource_filename(__name__, str(Path("example_data") / "registry.txt"))
)

with open(
    resource_filename(__name__, str(Path("example_data") / "index.json")), "r"
) as fdesc:
    __TRACKMAP = json.load(fdesc)


@deprecate_positional_args
def example(key, *, hq=False):
    """Retrieve the example recording identified by 'key'.

    The first time an example is requested, it will be downloaded from
    the remote repository over HTTPS.
    All subsequent requests will use a locally cached copy of the recording.

    For a list of examples (and their keys), see `librosa.util.list_examples`.

    By default, local files will be cached in the directory given by
    `pooch.os_cache('librosa')`.  You can override this by setting
    an environment variable ``LIBROSA_DATA_DIR`` prior to importing librosa:

    >>> import os
    >>> os.environ['LIBROSA_DATA_DIR'] = '/path/to/store/data'
    >>> import librosa

    Parameters
    ----------
    key : str
        The identifier for the track to load
    hq : bool
        If ``True``, return the high-quality version of the recording.
        If ``False``, return the 22KHz mono version of the recording.

    Returns
    -------
    path : str
        The path to the requested example file

    Examples
    --------
    Load "Hungarian Dance #5" by Johannes Brahms

    >>> y, sr = librosa.load(librosa.example('brahms'))

    Load "Vibe Ace" by Kevin MacLeod (the example previously packaged with librosa)
    in high-quality mode

    >>> y, sr = librosa.load(librosa.example('vibeace', hq=True))

    See Also
    --------
    librosa.util.list_examples
    pooch.os_cache
    """

    if key not in __TRACKMAP:
        raise ParameterError("Unknown example key: {}".format(key))

    if hq:
        ext = ".hq.ogg"
    else:
        ext = ".ogg"

    return __GOODBOY.fetch(__TRACKMAP[key]["path"] + ext)


ex = example
"""Alias for example"""


def list_examples():
    """List the available audio recordings included with librosa.

    Each recording is given a unique identifier (e.g., "brahms" or "nutcracker"),
    listed in the first column of the output.

    A brief description is provided in the second column.

    See Also
    --------
    util.example
    util.example_info
    """
    print("AVAILABLE EXAMPLES")
    print("-" * 68)
    for key in sorted(__TRACKMAP.keys()):
        print("{:10}\t{}".format(key, __TRACKMAP[key]["desc"]))


def example_info(key):
    """Display licensing and metadata information for the given example recording.

    The first time an example is requested, it will be downloaded from
    the remote repository over HTTPS.
    All subsequent requests will use a locally cached copy of the recording.

    For a list of examples (and their keys), see `librosa.util.list_examples`.

    By default, local files will be cached in the directory given by
    `pooch.os_cache('librosa')`.  You can override this by setting
    an environment variable ``LIBROSA_DATA_DIR`` prior to importing librosa.

    Parameters
    ----------
    key : str
        The identifier for the recording (see `list_examples`)

    See Also
    --------
    librosa.util.example
    librosa.util.list_examples
    pooch.os_cache
    """

    if key not in __TRACKMAP:
        raise ParameterError("Unknown example key: {}".format(key))

    license = __GOODBOY.fetch(__TRACKMAP[key]["path"] + ".txt")

    with open(license, "r") as fdesc:
        print("{:10s}\t{:s}".format(key, __TRACKMAP[key]["desc"]))
        print("-" * 68)
        for line in fdesc:
            print(line)


@deprecate_positional_args
def find_files(
    directory, *, ext=None, recurse=True, case_sensitive=False, limit=None, offset=0
):
    """Get a sorted list of (audio) files in a directory or directory sub-tree.

    Examples
    --------
    >>> # Get all audio files in a directory sub-tree
    >>> files = librosa.util.find_files('~/Music')

    >>> # Look only within a specific directory, not the sub-tree
    >>> files = librosa.util.find_files('~/Music', recurse=False)

    >>> # Only look for mp3 files
    >>> files = librosa.util.find_files('~/Music', ext='mp3')

    >>> # Or just mp3 and ogg
    >>> files = librosa.util.find_files('~/Music', ext=['mp3', 'ogg'])

    >>> # Only get the first 10 files
    >>> files = librosa.util.find_files('~/Music', limit=10)

    >>> # Or last 10 files
    >>> files = librosa.util.find_files('~/Music', offset=-10)

    Parameters
    ----------
    directory : str
        Path to look for files

    ext : str or list of str
        A file extension or list of file extensions to include in the search.

        Default: ``['aac', 'au', 'flac', 'm4a', 'mp3', 'ogg', 'wav']``

    recurse : boolean
        If ``True``, then all subfolders of ``directory`` will be searched.

        Otherwise, only ``directory`` will be searched.

    case_sensitive : boolean
        If ``False``, files matching upper-case version of
        extensions will be included.

    limit : int > 0 or None
        Return at most ``limit`` files. If ``None``, all files are returned.

    offset : int
        Return files starting at ``offset`` within the list.

        Use negative values to offset from the end of the list.

    Returns
    -------
    files : list of str
        The list of audio files.
    """

    if ext is None:
        ext = ["aac", "au", "flac", "m4a", "mp3", "ogg", "wav"]

    elif isinstance(ext, str):
        ext = [ext]

    # Cast into a set
    ext = set(ext)

    # Generate upper-case versions
    if not case_sensitive:
        # Force to lower-case
        ext = set([e.lower() for e in ext])
        # Add in upper-case versions
        ext |= set([e.upper() for e in ext])

    files = set()

    if recurse:
        for walk in os.walk(directory):
            files |= __get_files(walk[0], ext)
    else:
        files = __get_files(directory, ext)

    files = list(files)
    files.sort()
    files = files[offset:]
    if limit is not None:
        files = files[:limit]

    return files


def __get_files(dir_name, extensions):
    """Helper function to get files in a single directory"""

    # Expand out the directory
    dir_name = os.path.abspath(os.path.expanduser(dir_name))

    myfiles = set()

    for sub_ext in extensions:
        globstr = os.path.join(dir_name, "*" + os.path.extsep + sub_ext)
        myfiles |= set(glob.glob(globstr))

    return myfiles
