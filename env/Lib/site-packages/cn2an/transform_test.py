import unittest

from .transform import Transform


class TransformTest(unittest.TestCase):
    def setUp(self) -> None:
        self.strict_data_dict = {
            "小王捡了100块钱": "小王捡了一百块钱",
            "用户增长最快的3个城市": "用户增长最快的三个城市",
            "小王的生日是2001年3月4日": "小王的生日是二零零一年三月四日",
            "小王的生日是2012年12月12日": "小王的生日是二零一二年十二月十二日",
            "今天股价上涨了8%": "今天股价上涨了百分之八",
            "第2天股价下降了-3.8%": "第二天股价下降了百分之负三点八",
            "抛出去的硬币为正面的概率是1/2": "抛出去的硬币为正面的概率是二分之一",
            "现在室内温度为39℃，很热啊！": "现在室内温度为三十九摄氏度，很热啊！",
            "创业板指9月9日早盘低开1.57%": "创业板指九月九日早盘低开百分之一点五七"
        }

        self.smart_data_dict = {
            "约2.5亿年~6500万年": "约250000000年~65000000年",
            "廿二日，日出东方": "22日，日出东方",
            "大陆": "大陆",
            "半斤": "0.5斤",
            "两个": "2个",
        }

        self.t = Transform()

    def test_transform(self) -> None:
        for strict_item in self.strict_data_dict.keys():
            self.assertEqual(self.t.transform(strict_item, "an2cn"), self.strict_data_dict[strict_item])
            self.assertEqual(self.t.transform(self.strict_data_dict[strict_item], "cn2an"), strict_item)

        for smart_item in self.smart_data_dict.keys():
            self.assertEqual(self.t.transform(smart_item, "cn2an"), self.smart_data_dict[smart_item])


if __name__ == '__main__':
    unittest.main()
