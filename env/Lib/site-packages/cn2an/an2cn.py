from typing import Union
from warnings import warn

from proces import preprocess

from .conf import NUMBER_LOW_AN2CN, NUMBER_UP_AN2CN, UNIT_LOW_ORDER_AN2CN, UNIT_UP_ORDER_AN2CN


class An2Cn(object):
    def __init__(self) -> None:
        self.all_num = "0123456789"
        self.number_low = NUMBER_LOW_AN2CN
        self.number_up = NUMBER_UP_AN2CN
        self.mode_list = ["low", "up", "rmb", "direct"]

    def an2cn(self, inputs: Union[str, int, float] = None, mode: str = "low") -> str:
        """阿拉伯数字转中文数字

        :param inputs: 阿拉伯数字
        :param mode: low 小写数字，up 大写数字，rmb 人民币大写，direct 直接转化
        :return: 中文数字
        """
        if inputs is not None and inputs != "":
            if mode not in self.mode_list:
                raise ValueError(f"mode 仅支持 {str(self.mode_list)} ！")

            # 将数字转化为字符串，这里会有Python会自动做转化
            # 1. -> 1.0 1.00 -> 1.0 -0 -> 0
            if not isinstance(inputs, str):
                inputs = self.__number_to_string(inputs)

            # 数据预处理：
            # 1. 繁体转简体
            # 2. 全角转半角
            inputs = preprocess(inputs, pipelines=[
                "traditional_to_simplified",
                "full_angle_to_half_angle"
            ])

            # 检查数据是否有效
            self.__check_inputs_is_valid(inputs)

            # 判断正负
            if inputs[0] == "-":
                sign = "负"
                inputs = inputs[1:]
            else:
                sign = ""

            if mode == "direct":
                output = self.__direct_convert(inputs)
            else:
                # 切割整数部分和小数部分
                split_result = inputs.split(".")
                len_split_result = len(split_result)
                if len_split_result == 1:
                    # 不包含小数的输入
                    integer_data = split_result[0]
                    if mode == "rmb":
                        output = self.__integer_convert(integer_data, "up") + "元整"
                    else:
                        output = self.__integer_convert(integer_data, mode)
                elif len_split_result == 2:
                    # 包含小数的输入
                    integer_data, decimal_data = split_result
                    if mode == "rmb":
                        int_data = self.__integer_convert(integer_data, "up")
                        dec_data = self.__decimal_convert(decimal_data, "up")
                        len_dec_data = len(dec_data)

                        if len_dec_data == 0:
                            output = int_data + "元整"
                        elif len_dec_data == 1:
                            raise ValueError(f"异常输出：{dec_data}")
                        elif len_dec_data == 2:
                            if dec_data[1] != "零":
                                if int_data == "零":
                                    output = dec_data[1] + "角"
                                else:
                                    output = int_data + "元" + dec_data[1] + "角"
                            else:
                                output = int_data + "元整"
                        else:
                            if dec_data[1] != "零":
                                if dec_data[2] != "零":
                                    if int_data == "零":
                                        output = dec_data[1] + "角" + dec_data[2] + "分"
                                    else:
                                        output = int_data + "元" + dec_data[1] + "角" + dec_data[2] + "分"
                                else:
                                    if int_data == "零":
                                        output = dec_data[1] + "角"
                                    else:
                                        output = int_data + "元" + dec_data[1] + "角"
                            else:
                                if dec_data[2] != "零":
                                    if int_data == "零":
                                        output = dec_data[2] + "分"
                                    else:
                                        output = int_data + "元" + "零" + dec_data[2] + "分"
                                else:
                                    output = int_data + "元整"
                    else:
                        output = self.__integer_convert(integer_data, mode) + self.__decimal_convert(decimal_data, mode)
                else:
                    raise ValueError(f"输入格式错误：{inputs}！")
        else:
            raise ValueError("输入数据为空！")

        return sign + output

    def __direct_convert(self, inputs: str) -> str:
        _output = ""
        for d in inputs:
            if d == ".":
                _output += "点"
            else:
                _output += self.number_low[int(d)]
        return _output

    @staticmethod
    def __number_to_string(number_data: Union[int, float]) -> str:
        # 小数处理：python 会自动把 0.00005 转化成 5e-05，因此 str(0.00005) != "0.00005"
        string_data = str(number_data)
        if "e" in string_data:
            string_data_list = string_data.split("e")
            string_key = string_data_list[0]
            string_value = string_data_list[1]
            if string_value[0] == "-":
                string_data = "0." + "0" * (int(string_value[1:]) - 1) + string_key
            else:
                string_data = string_key + "0" * int(string_value)
        return string_data

    def __check_inputs_is_valid(self, check_data: str) -> None:
        # 检查输入数据是否在规定的字典中
        all_check_keys = self.all_num + ".-"
        for data in check_data:
            if data not in all_check_keys:
                raise ValueError(f"输入的数据不在转化范围内：{data}！")

    def __integer_convert(self, integer_data: str, mode: str) -> str:
        if mode == "low":
            numeral_list = NUMBER_LOW_AN2CN
            unit_list = UNIT_LOW_ORDER_AN2CN
        elif mode == "up":
            numeral_list = NUMBER_UP_AN2CN
            unit_list = UNIT_UP_ORDER_AN2CN
        else:
            raise ValueError(f"error mode: {mode}")

        # 去除前面的 0，比如 007 => 7
        integer_data = str(int(integer_data))

        len_integer_data = len(integer_data)
        if len_integer_data > len(unit_list):
            raise ValueError(f"超出数据范围，最长支持 {len(unit_list)} 位")

        output_an = ""
        for i, d in enumerate(integer_data):
            if int(d):
                output_an += numeral_list[int(d)] + unit_list[len_integer_data - i - 1]
            else:
                if not (len_integer_data - i - 1) % 4:
                    output_an += numeral_list[int(d)] + unit_list[len_integer_data - i - 1]

                if i > 0 and not output_an[-1] == "零":
                    output_an += numeral_list[int(d)]

        output_an = output_an.replace("零零", "零").replace("零万", "万").replace("零亿", "亿").replace("亿万", "亿") \
            .strip("零")

        # 解决「一十几」问题
        if output_an[:2] in ["一十"]:
            output_an = output_an[1:]

        # 0 - 1 之间的小数
        if not output_an:
            output_an = "零"

        return output_an

    def __decimal_convert(self, decimal_data: str, o_mode: str) -> str:
        len_decimal_data = len(decimal_data)

        if len_decimal_data > 16:
            warn(f"注意：小数部分长度为 {len_decimal_data} ，将自动截取前 16 位有效精度！")
            decimal_data = decimal_data[:16]

        if len_decimal_data:
            output_an = "点"
        else:
            output_an = ""

        if o_mode == "low":
            numeral_list = NUMBER_LOW_AN2CN
        elif o_mode == "up":
            numeral_list = NUMBER_UP_AN2CN
        else:
            raise ValueError(f"error mode: {o_mode}")

        for data in decimal_data:
            output_an += numeral_list[int(data)]
        return output_an
