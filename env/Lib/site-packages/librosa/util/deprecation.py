#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Deprecation utilities"""

import inspect
import warnings


class Deprecated(object):
    """A dummy class to catch usage of deprecated variable names"""

    def __repr__(self):
        return "<DEPRECATED parameter>"


def rename_kw(
    *, old_name, old_value, new_name, new_value, version_deprecated, version_removed
):
    """Handle renamed arguments.

    Parameters
    ----------
    old_name : str
    old_value
        The name and value of the old argument
    new_name : str
    new_value
        The name and value of the new argument
    version_deprecated : str
        The version at which the old name became deprecated
    version_removed : str
        The version at which the old name will be removed

    Returns
    -------
    value
        - ``new_value`` if ``old_value`` of type `Deprecated`
        - ``old_value`` otherwise

    Warnings
    --------
    if ``old_value`` is not of type `Deprecated`

    """
    if isinstance(old_value, Deprecated):
        return new_value
    else:
        stack = inspect.stack()
        dep_func = stack[1]

        warnings.warn(
            "{:s}() keyword argument '{:s}' has been "
            "renamed to '{:s}' in version {:}."
            "\n\tThis alias will be removed in version "
            "{:}.".format(
                dep_func[3], old_name, new_name, version_deprecated, version_removed
            ),
            category=DeprecationWarning,
            stacklevel=3,
        )

        return old_value
