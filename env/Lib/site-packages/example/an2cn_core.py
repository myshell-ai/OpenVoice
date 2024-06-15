from time import time

numeral_list = {0: "零",
                1: "一",
                2: "二",
                3: "三",
                4: "四",
                5: "五",
                6: "六",
                7: "七",
                8: "八",
                9: "九"}
unit_list = ["", "十", "百", "千", "万", "十", "百", "千", "亿", "十", "百", "千", "万", "十", "百", "千"]


def an2cn(integer_data: int) -> str:
    integer_data = str(integer_data)
    len_integer_data = len(integer_data)
    output_an = ""
    for i, d in enumerate(integer_data):
        if int(d):
            output_an += numeral_list[int(d)] + unit_list[len_integer_data - i - 1]
        else:
            if not (len_integer_data - i - 1) % 4:
                output_an += numeral_list[int(d)] + unit_list[len_integer_data - i - 1]

            if i > 0 and not output_an[-1] == "零":
                output_an += numeral_list[int(d)]

    output_an = output_an.replace("零零", "零").replace("零万", "万").replace("零亿", "亿").strip("零")
    return output_an


def run_an2cn_core_ten_thousand_times():
    t_start = time()
    for _ in range(1000000):
        assert an2cn(9876543298765432) == "九千八百七十六万五千四百三十二亿九千八百七十六万五千四百三十二"
    t_end = time()
    print(round(t_end - t_start, 3))


if __name__ == '__main__':
    run_an2cn_core_ten_thousand_times()
