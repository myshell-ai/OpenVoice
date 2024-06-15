import unittest

from .cn2an import Cn2An


class Cn2anTest(unittest.TestCase):
    def setUp(self) -> None:
        self.strict_data_dict = {
            "零": 0,
            "一": 1,
            "十": 10,
            "十一": 11,
            "一十一": 11,
            "二十": 20,
            "二十一": 21,
            "一百": 100,
            "一百零一": 101,
            "一百一十": 110,
            "一百一十一": 111,
            "一千": 1000,
            "一千一百": 1100,
            "一千一百一十": 1110,
            "一千一百一十一": 1111,
            "一千零一十": 1010,
            "一千零十": 1010,
            "一千零十一": 1011,
            "一千零一十一": 1011,
            "一千零一": 1001,
            "一千一百零一": 1101,
            "一万一千一百一十一": 11111,
            "一十一万一千一百一十一": 111111,
            "一百一十一万一千一百一十一": 1111111,
            "一千一百一十一万一千一百一十一": 11111111,
            "一亿一千一百一十一万一千一百一十一": 111111111,
            "一十一亿一千一百一十一万一千一百一十一": 1111111111,
            "一百一十一亿一千一百一十一万一千一百一十一": 11111111111,
            "一千一百一十一亿一千一百一十一万一千一百一十一": 111111111111,
            "一千一百一十一万一千一百一十一亿一千一百一十一万一千一百一十一": 1111111111111111,
            "壹": 1,
            "拾": 10,
            "拾壹": 11,
            "壹拾壹": 11,
            "壹佰壹拾壹": 111,
            "壹仟壹佰壹拾壹": 1111,
            "壹万壹仟壹佰壹拾壹": 11111,
            "壹拾壹万壹仟壹佰壹拾壹": 111111,
            "壹佰壹拾壹万壹仟壹佰壹拾壹": 1111111,
            "壹仟壹佰壹拾壹万壹仟壹佰壹拾壹": 11111111,
            "壹亿壹仟壹佰壹拾壹万壹仟壹佰壹拾壹": 111111111,
            "壹拾壹亿壹仟壹佰壹拾壹万壹仟壹佰壹拾壹": 1111111111,
            "壹佰壹拾壹亿壹仟壹佰壹拾壹万壹仟壹佰壹拾壹": 11111111111,
            "壹仟壹佰壹拾壹亿壹仟壹佰壹拾壹万壹仟壹佰壹拾壹": 111111111111,
            "壹拾壹元整": 11,
            "壹佰壹拾壹圆整": 111,
            "壹拾壹元正": 11,
            "壹拾壹圆正": 11,
            "壹拾壹元壹角": 11.1,
            "壹拾壹元壹角壹分": 11.11,
            "十万": 100000,
            "十万零一": 100001,
            "一万零一": 10001,
            "一万零一十一": 10011,
            "一万零一百一十一": 10111,
            "一万零百一十一": 10111,
            "一十万零一": 100001,
            "一百万零一": 1000001,
            "一千万零一": 10000001,
            "一千零一万一千零一": 10011001,
            "一千零一万零一": 10010001,
            "一亿零一": 100000001,
            "一十亿零一": 1000000001,
            "一百亿零一": 10000000001,
            "一千零一亿一千零一万一千零一": 100110011001,
            "一千亿一千万一千零一": 100010001001,
            "一千亿零一": 100000000001,
            "零点零零零零零零零零零零零零零零一": 0.000000000000001,
            "零点零零零零零零零零零零零零零一": 0.00000000000001,
            "零点零零零零零零零零零零零零一": 0.0000000000001,
            "零点零零零零零零零零零零零一": 0.000000000001,
            "零点零零零零零零零零零零一": 0.00000000001,
            "零点零零零零零零零零零一": 0.0000000001,
            "零点零零零零零零零零一": 0.000000001,
            "零点零零零零零零零一": 0.00000001,
            "零点零零零零零零一": 0.0000001,
            "零点零零零零零一": 0.000001,
            "零点零零零零一": 0.00001,
            "零点零零零一": 0.0001,
            "零点零零一": 0.001,
            "零点零一": 0.01,
            "零点一": 0.1,
            "负一": -1,
            "负二": -2,
            "负十": -10,
            "负十一": -11,
            "负一十一": -11,
            # 古语
            "廿二": 22,
        }

        self.normal_data_dict = {
            "一一": 11,
            "一一一": 111,
            "壹壹": 11,
            "壹壹壹": 111,
            "零点零": 0,
            "零点零零": 0,
            "一七二零": 1720,
            "一七二零点一": 1720.1,
            "一七二零点一三四": 1720.134,
            "一二三": 123,
            "负零点一零": -0.1,
            "负一七二零": -1720,
            "负一七二零点一": -1720.1,
            # 口语
            "三万五": 35000,
            "十三万五": 135000,
            "两千六": 2600,
            "一百二": 120,
            "一百二十万三": 1203000,
            # 繁体
            "兩千六": 2600,
            # 大写
            "壹拾壹元": 11,
            "壹佰壹拾壹圆": 111,
            "壹拾壹圆": 11,
            # 特殊
            "〇": 0,
        }

        self.smart_data_dict = {
            "100万": 1000000,
            "100万三千": 1003000,
            "200亿零四千230": 20000004230,
            "一百点123": 100.123,
            "10.1万": 101000,
            "-10.1万": -101000,
            "35.1亿": 3510000000,
            "10.1": 10.1,
            "-10.1": -10.1,
        }

        self.error_smart_datas = [
            "10.1万零100",
            "10..1万",
        ]

        self.error_normal_datas = [
            "零点",
            "点零",
            "零点点",
            "零点零大",
        ]
        self.error_normal_datas.extend(self.error_smart_datas)
        self.error_normal_datas.extend(list(self.smart_data_dict.keys()))

        self.error_strict_datas = [
            "一一",
            "壹壹",
            "零点",
            "点零",
            "点一",
            "百十一",
            "十一十二",
            "负十一十二",
            "十七十八",
        ]
        self.error_strict_datas.extend(self.error_normal_datas)
        self.error_strict_datas.extend(list(self.normal_data_dict.keys()))

        # 不可修改位置
        self.normal_data_dict.update(self.strict_data_dict)
        self.smart_data_dict.update(self.normal_data_dict)

        self.ca = Cn2An()

    def test_cn2an(self) -> None:
        for strict_item in self.strict_data_dict.keys():
            self.assertEqual(self.ca.cn2an(strict_item, "strict"),
                             self.strict_data_dict[strict_item])

        for normal_item in self.normal_data_dict.keys():
            self.assertEqual(self.ca.cn2an(normal_item, "normal"),
                             self.normal_data_dict[normal_item])

        for smart_item in self.smart_data_dict.keys():
            self.assertEqual(self.ca.cn2an(smart_item, "smart"),
                             self.smart_data_dict[smart_item])

        for error_strict_item in self.error_strict_datas:
            try:
                self.ca.cn2an(error_strict_item)
            except ValueError as e:
                self.assertEqual(type(e), ValueError)
            else:
                raise Exception(f'ValueError not raised: {error_strict_item}')

        for error_normal_item in self.error_normal_datas:
            try:
                self.ca.cn2an(error_normal_item)
            except ValueError as e:
                self.assertEqual(type(e), ValueError)
            else:
                raise Exception(f'ValueError not raised: {error_normal_item}')

        for error_smart_item in self.error_smart_datas:
            try:
                self.ca.cn2an(error_smart_item)
            except ValueError as e:
                self.assertEqual(type(e), ValueError)
            else:
                raise Exception(f'ValueError not raised: {error_smart_item}')


if __name__ == '__main__':
    unittest.main()
