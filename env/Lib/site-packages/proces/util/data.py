from typing import Set, Pattern
from re import compile

from proces.data.province_city import all_cities_data


def get_all_cities() -> Set:
    cities = set()
    for province, p_value in all_cities_data.items():
        cities.add(province)
        for city in p_value:
            cities.add(f"{province}/{city}")
    return cities


def get_city_pattern() -> Pattern[str]:
    province_cache = {}
    special_province_map = {
        "新疆维吾尔": "新疆(维吾尔)?",
        "宁夏回族": "宁夏(回族)?",
        "广西壮族": "广西(壮族)?",
    }

    def get_province_item_unit(province_data):
        if province_data not in province_cache.keys():
            if province_data[-5:] in ["特别行政区"]:
                item, unit = province_data[:-5], province_data[-5:]
            elif province_data[-3:] in ["自治区"]:
                if province_data[:-3] in special_province_map.keys():
                    item, unit = special_province_map[province_data[:-3]], province_data[-3:]
                else:
                    item, unit = province_data[:-3], province_data[-3:]
            elif province_data[-1] in ["省", "市"]:
                item, unit = province_data[:-1], province_data[-1]
            else:
                raise Exception(province_data)

            province_cache[province_data] = (item, unit)
        else:
            item, unit = province_cache[province_data]
        return item, unit

    city_cache = {}
    special_city_map = {
        "临夏回族": "临夏(回族)?",
        "黔南布依族苗族": "黔南(布依族)?(苗族)?",
        "黔东南苗族侗族": "黔东南(苗族)?(侗族)?",
        "红河哈尼族彝族": "红河(哈尼族)?(彝族)?",
        "恩施土家族苗族": "恩施(土家族)?(苗族)?",
        "延边朝鲜族": "延边(朝鲜族)?",
        "巴音郭楞蒙古": "巴音郭楞(蒙古)?",
        "白沙黎族": "白沙(黎族)?",
        "黄南藏族": "黄南(藏族)?",
        "琼中黎族苗族": "琼中(黎族)?(苗族)?",
        "大理白族": "大理(白族)?",
        "保亭黎族苗族": "保亭(黎族)?(苗族)?",
        "湘西土家族苗族": "湘西(土家族)?(苗族)?",
        "海北藏族": "海北(藏族)?",
        "怒江傈僳族": "怒江(傈僳族)?",
        "阿坝藏族羌族": "阿坝(藏族)?(羌族)?",
        "果洛藏族": "果洛(藏族)?",
        "海西蒙古族藏族": "海西(蒙古族)?(藏族)?",
        "玉树藏族": "玉树(藏族)?",
        "博尔塔拉蒙古": "博尔塔拉(蒙古)?",
        "甘孜藏族": "甘孜(藏族)?",
        "迪庆藏族": "迪庆(藏族)?",
        "德宏傣族景颇族": "德宏(傣族)?(景颇族)?",
        "昌吉回族": "昌吉(回族)?",
        "伊犁哈萨克": "伊犁(哈萨克)?",
        "昌江黎族": "昌江(黎族)?",
        "凉山彝族": "凉山(彝族)?",
        "陵水黎族": "陵水(黎族)?",
        "海南藏族": "海南(藏族)?",
        "西双版纳傣族": "西双版纳(傣族)?",
        "克孜勒苏柯尔克孜": "克孜勒苏(柯尔克孜)?",
        "甘南藏族": "甘南(藏族)?",
        "文山壮族苗族": "文山(壮族)?(苗族)?",
        "黔西南布依族苗族": "黔西南(布依族)?(苗族)?",
        "乐东黎族": "乐东(黎族)?",
        "楚雄彝族": "楚雄(彝族)?",
    }

    def get_city_item_unit(city_data):
        if city_data not in city_cache.keys():
            if city_data[-3:] in ["自治县", "自治州"]:
                if city_data[:-3] in special_city_map.keys():
                    item, unit = special_city_map[city_data[:-3]], f"(自治)?{city_data[-1]}"
                else:
                    item, unit = city_data[:-3], f"(自治)?{city_data[-1]}"
            elif city_data[-2:] in ["林区", "地区"]:
                item, unit = city_data[:-2], city_data[-2:]
            elif city_data[-1] in ["县", "市", "盟"]:
                item, unit = city_data[:-1], city_data[-1]
            else:
                raise Exception(city_data)
            city_cache[city_data] = (item, unit)
        else:
            item, unit = city_cache[city_data]
        return item, unit

    patterns = set()
    for line in get_all_cities():
        if "/" in line:
            province, city = line.split("/")
            p_item, p_unit = get_province_item_unit(province)
            c_item, c_unit = get_city_item_unit(city)
            patterns.add(f"{p_item}({p_unit})?{c_item}({c_unit})?")
            patterns.add(f"{c_item}({c_unit})?")
        else:
            province = line
            p_item, p_unit = get_province_item_unit(province)
            patterns.add(f"{p_item}({p_unit})?")

    patterns = sorted(patterns, key=lambda x: len(x), reverse=True)
    ptn_str = "|".join(patterns)
    return compile(f"({ptn_str})")
