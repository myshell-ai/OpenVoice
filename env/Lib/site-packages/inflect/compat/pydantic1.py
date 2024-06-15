try:
    from pydantic import validate_call  # type: ignore
except ImportError:
    # Pydantic 1
    from pydantic import validate_arguments as validate_call  # type: ignore


__all__ = ['validate_call']
