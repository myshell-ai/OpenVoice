__version__ = "0.5.22"

from .cn2an import Cn2An
from .an2cn import An2Cn
from .transform import Transform

cn2an = Cn2An().cn2an
an2cn = An2Cn().an2cn
transform = Transform().transform

__all__ = [
    "__version__",
    "cn2an",
    "an2cn",
    "transform"
]
