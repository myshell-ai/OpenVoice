import torbjorn as tbn

from .an2cn import An2Cn
from .cn2an import Cn2An

ac = An2Cn()
ca = Cn2An()

an = 9876543298765432
cn = "九千八百七十六万五千四百三十二亿九千八百七十六万五千四百三十二"


@tbn.run_time
def run_cn2an_ten_thousand_times() -> None:
    for _ in range(10000):
        result = ca.cn2an(cn)
        assert result == an


@tbn.run_time
def run_an2cn_ten_thousand_times() -> None:
    for _ in range(10000):
        result = ac.an2cn(an)
        assert result == cn


if __name__ == '__main__':
    run_cn2an_ten_thousand_times()
    run_an2cn_ten_thousand_times()
