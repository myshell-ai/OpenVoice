import functools
import warnings


class AVDeprecationWarning(DeprecationWarning):
    pass


class AttributeRenamedWarning(AVDeprecationWarning):
    pass


class MethodDeprecationWarning(AVDeprecationWarning):
    pass


# DeprecationWarning is not printed by default (unless in __main__). We
# really want these to be seen, but also to use the "correct" base classes.
# So we're putting a filter in place to show our warnings. The users can
# turn them back off if they want.
warnings.filterwarnings("default", "", AVDeprecationWarning)


class renamed_attr(object):

    """Proxy for renamed attributes (or methods) on classes.
    Getting and setting values will be redirected to the provided name,
    and warnings will be issues every time.

    """

    def __init__(self, new_name):
        self.new_name = new_name
        self._old_name = None

    def old_name(self, cls):
        if self._old_name is None:
            for k, v in vars(cls).items():
                if v is self:
                    self._old_name = k
                    break
        return self._old_name

    def __get__(self, instance, cls):
        old_name = self.old_name(cls)
        warnings.warn(
            "{0}.{1} is deprecated; please use {0}.{2}.".format(
                cls.__name__,
                old_name,
                self.new_name,
            ),
            AttributeRenamedWarning,
            stacklevel=2,
        )
        return getattr(instance if instance is not None else cls, self.new_name)

    def __set__(self, instance, value):
        old_name = self.old_name(instance.__class__)
        warnings.warn(
            "{0}.{1} is deprecated; please use {0}.{2}.".format(
                instance.__class__.__name__,
                old_name,
                self.new_name,
            ),
            AttributeRenamedWarning,
            stacklevel=2,
        )
        setattr(instance, self.new_name, value)


class method(object):
    def __init__(self, func):
        functools.update_wrapper(self, func, ("__name__", "__doc__"))
        self.func = func

    def __get__(self, instance, cls):
        warning = MethodDeprecationWarning(
            "{}.{} is deprecated.".format(cls.__name__, self.func.__name__)
        )
        warnings.warn(warning, stacklevel=2)
        return self.func.__get__(instance, cls)
