# 数字映射
number_map = {
    "零": 0,
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9
}

# 单位映射
unit_map = {
    "十": 10,
    "百": 100,
    "千": 1000,
    "万": 10000,
    "亿": 100000000
}


# 正向遍历 1
def forward_cn2an_one(inputs):
    output = 0
    unit = 1
    num = 0
    for index, cn_num in enumerate(inputs):
        if cn_num in number_map:
            # 数字
            num = number_map[cn_num]
            # 最后的个位数字
            if index == len(inputs) - 1:
                output = output + num
        elif cn_num in unit_map:
            # 单位
            unit = unit_map[cn_num]
            # 累加
            output = output + num * unit
            num = 0
        else:
            raise ValueError(f"{cn_num} 不在转化范围内")

    return output


# 正向遍历 2
def forward_cn2an_two(inputs):
    output = 0
    unit = 1
    num = 0
    for index, cn_num in enumerate(inputs):
        if cn_num in number_map:
            # 数字
            num = number_map[cn_num]
            # 最后的个位数字
            if index == len(inputs) - 1:
                output = output + num
        elif cn_num in unit_map:
            # 单位
            unit = unit_map[cn_num]
            # 判断出万、亿，先把前面的累加再乘以单位万、亿
            if unit % 10000 == 0:
                output = (output + num) * unit
            else:
                # 累加
                output = output + num * unit
            num = 0
        else:
            raise ValueError(f"{cn_num} 不在转化范围内")

    return output


# 正向遍历 3
def forward_cn2an_three(inputs):
    output = 0
    unit = 1
    num = 0
    # 亿位以上的输出
    hundred_million_output = 0
    for index, cn_num in enumerate(inputs):
        if cn_num in number_map:
            # 数字
            num = number_map[cn_num]
            # 最后的个位数字
            if index == len(inputs) - 1:
                # 把亿位和中间输出以及个位上的一起加起来
                output = hundred_million_output + output + num
        elif cn_num in unit_map:
            # 单位
            unit = unit_map[cn_num]
            # 判断出万，前面的累加再乘以单位万
            if unit == 10000:
                output = (output + num) * unit
            # 判断出亿，前面累加乘以亿后赋值给 hundred_million_output， output 重置为 0
            elif unit == 100000000:
                hundred_million_output = (output + num) * unit
                output = 0
            else:
                # 累加
                output = output + num * unit
            num = 0
        else:
            raise ValueError(f"{cn_num} 不在转化范围内")

    return output


# 反向遍历 1
def backward_cn2an_one(inputs):
    output = 0
    unit = 1
    num = 0
    for index, cn_num in enumerate(reversed(inputs)):
        if cn_num in number_map:
            # 数字
            num = number_map[cn_num]
            # 累加
            output = output + num * unit
        elif cn_num in unit_map:
            # 单位
            unit = unit_map[cn_num]
        else:
            raise ValueError(f"{cn_num} 不在转化范围内")

    return output


# 反向遍历 2
def backward_cn2an_two(inputs):
    output = 0
    unit = 1
    # 万、亿的单位
    ten_thousand_unit = 1
    num = 0
    for index, cn_num in enumerate(reversed(inputs)):
        if cn_num in number_map:
            # 数字
            num = number_map[cn_num]
            # 累加
            output = output + num * unit
        elif cn_num in unit_map:
            # 单位
            unit = unit_map[cn_num]
            # 判断出万、亿
            if unit % 10000 == 0:
                ten_thousand_unit = unit

            if unit < ten_thousand_unit:
                unit = ten_thousand_unit * unit
        else:
            raise ValueError(f"{cn_num} 不在转化范围内")

    return output


# 反向遍历 3
def backward_cn2an_three(inputs):
    output = 0
    unit = 1
    # 万、亿的单位
    ten_thousand_unit = 1
    num = 0
    for index, cn_num in enumerate(reversed(inputs)):
        if cn_num in number_map:
            # 数字
            num = number_map[cn_num]
            # 累加
            output = output + num * unit
        elif cn_num in unit_map:
            # 单位
            unit = unit_map[cn_num]
            # 判断出万、亿
            if unit % 10000 == 0:
                # 万、亿
                if unit > ten_thousand_unit:
                    ten_thousand_unit = unit
                # 万亿
                else:
                    ten_thousand_unit = unit * ten_thousand_unit
                    unit = ten_thousand_unit

            if unit < ten_thousand_unit:
                unit = ten_thousand_unit * unit
        else:
            raise ValueError(f"{cn_num} 不在转化范围内")

    return output


if __name__ == '__main__':
    cn_data1 = ["一百二十三", 123]
    cn_data2 = ["一千二百三十四万五千六百七十八", 12345678]
    cn_data3 = ["一亿二千三百四十五万六千七百八十一", 123456781]
    cn_data4 = ["一千二百三十四万五千六百七十八亿一千二百三十四万五千六百七十八", 1234567812345678]

    # 正向遍历 1，用数字乘单位，然后直接把他们累加起来
    print("\n# forward cn2an 1")
    print(cn_data1[0], forward_cn2an_one(cn_data1[0]), forward_cn2an_one(cn_data1[0]) == cn_data1[1])
    print(cn_data2[0], forward_cn2an_one(cn_data2[0]), forward_cn2an_one(cn_data2[0]) == cn_data2[1])
    # # forward cn2an 1
    # 一百二十三 123 True
    # 一千二百三十四万五千六百七十八 46908 False

    # 正向遍历 2，用数字乘单位，碰到万的时候先把前面的累加起来乘以万，再直接把后面的累加起来
    print("\n# forward cn2an 2")
    print(cn_data1[0], forward_cn2an_two(cn_data1[0]), forward_cn2an_two(cn_data1[0]) == cn_data1[1])
    print(cn_data2[0], forward_cn2an_two(cn_data2[0]), forward_cn2an_two(cn_data2[0]) == cn_data2[1])
    print(cn_data3[0], forward_cn2an_two(cn_data3[0]), forward_cn2an_two(cn_data3[0]) == cn_data3[1])
    # # forward cn2an 2
    # 一百二十三 123 True
    # 一千二百三十四万五千六百七十八 12345678 True
    # 一亿二千三百四十五万六千七百八十一 1000023456781 False

    # 正向遍历 3，用数字乘单位，碰到亿的时候先把前面的累加起来乘以亿，碰到万的时候先把前面的累加起来乘以万，再直接把后面的累加起来
    # 最后把亿位加上后面的累加和
    print("\n# forward cn2an 3")
    print(cn_data1[0], forward_cn2an_three(cn_data1[0]), forward_cn2an_three(cn_data1[0]) == cn_data1[1])
    print(cn_data2[0], forward_cn2an_three(cn_data2[0]), forward_cn2an_three(cn_data2[0]) == cn_data2[1])
    print(cn_data3[0], forward_cn2an_three(cn_data3[0]), forward_cn2an_three(cn_data3[0]) == cn_data3[1])
    print(cn_data4[0], forward_cn2an_three(cn_data4[0]), forward_cn2an_three(cn_data4[0]) == cn_data4[1])
    # # forward cn2an 3
    # 一百二十三 123 True
    # 一千二百三十四万五千六百七十八 12345678 True
    # 一亿二千三百四十五万六千七百八十一 123456781 True
    # 一千二百三十四万五千六百七十八亿一千二百三十四万五千六百七十八 1234567812345678 True

    # 反向遍历 1，用数字乘单位，然后直接把他们累加起来
    print("\n# backward cn2an 1")
    print(cn_data1[0], backward_cn2an_one(cn_data1[0]), backward_cn2an_one(cn_data1[0]) == cn_data1[1])
    print(cn_data2[0], backward_cn2an_one(cn_data2[0]), backward_cn2an_one(cn_data2[0]) == cn_data2[1])
    # backward cn2an 1
    # 一百二十三 123 True
    # 一千二百三十四万五千六百七十八 46908 False

    # 反向遍历 2，用数字乘单位，然后直接把他们累加起来，碰到万或亿的时候把单位乘万或亿
    print("\n# backward cn2an 2")
    print(cn_data1[0], backward_cn2an_two(cn_data1[0]), backward_cn2an_two(cn_data1[0]) == cn_data1[1])
    print(cn_data2[0], backward_cn2an_two(cn_data2[0]), backward_cn2an_two(cn_data2[0]) == cn_data2[1])
    print(cn_data3[0], backward_cn2an_two(cn_data3[0]), backward_cn2an_two(cn_data3[0]) == cn_data3[1])
    print(cn_data4[0], backward_cn2an_two(cn_data4[0]), backward_cn2an_two(cn_data4[0]) == cn_data4[1])
    # # backward cn2an 2
    # 一百二十三 123 True
    # 一千二百三十四万五千六百七十八 12345678 True
    # 一亿二千三百四十五万六千七百八十一 123456781 True
    # 一千二百三十四万五千六百七十八亿一千二百三十四万五千六百七十八 567824685678 False

    # 反向遍历 3，用数字乘单位，然后直接把他们累加起来，碰到万或亿的时候把单位乘万或亿，碰到万亿时把单位乘万亿
    print("\n# backward cn2an 3")
    print(cn_data1[0], backward_cn2an_three(cn_data1[0]), backward_cn2an_three(cn_data1[0]) == cn_data1[1])
    print(cn_data2[0], backward_cn2an_three(cn_data2[0]), backward_cn2an_three(cn_data2[0]) == cn_data2[1])
    print(cn_data3[0], backward_cn2an_three(cn_data3[0]), backward_cn2an_three(cn_data3[0]) == cn_data3[1])
    print(cn_data4[0], backward_cn2an_three(cn_data4[0]), backward_cn2an_three(cn_data4[0]) == cn_data4[1])
    # # backward cn2an 3
    # 一百二十三 123 True
    # 一千二百三十四万五千六百七十八 12345678 True
    # 一亿二千三百四十五万六千七百八十一 123456781 True
    # 一千二百三十四万五千六百七十八亿一千二百三十四万五千六百七十八 1234567812345678 True
