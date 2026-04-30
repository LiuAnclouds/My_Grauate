from __future__ import annotations

import hashlib
import random


FAMILY_NAMES = ["李", "王", "张", "刘", "陈", "杨", "黄", "赵", "吴", "周", "徐", "孙", "朱", "胡"]
GIVEN_NAMES = ["明轩", "雨桐", "子涵", "嘉宁", "思远", "若琳", "浩然", "佳怡", "俊杰", "诗涵", "晨阳", "欣悦"]
REGIONS = ["上海市浦东新区", "北京市朝阳区", "杭州市西湖区", "深圳市南山区", "成都市锦江区", "南京市玄武区", "武汉市江汉区", "苏州市工业园区"]
OCCUPATIONS = ["个体经营者", "公司职员", "网约车司机", "电商商户", "财务人员", "销售人员", "自由职业者", "物流从业者"]


def synthetic_person_for_node(node_id: str) -> dict[str, str]:
    seed = int(hashlib.sha256(str(node_id).encode("utf-8")).hexdigest()[:12], 16)
    rng = random.Random(seed)
    display_name = f"{rng.choice(FAMILY_NAMES)}{rng.choice(GIVEN_NAMES)}"
    area = rng.choice(["110101", "310101", "330102", "440305", "510104", "320102"])
    birth_year = rng.randint(1970, 2004)
    birth_month = rng.randint(1, 12)
    birth_day = rng.randint(1, 28)
    sequence = rng.randint(100, 999)
    id_number = f"{area}{birth_year:04d}{birth_month:02d}{birth_day:02d}{sequence:03d}{rng.randint(0, 9)}"
    phone = f"1{rng.choice([3, 5, 7, 8, 9])}{rng.randint(100000000, 999999999)}"
    return {
        "display_name": display_name,
        "id_number": id_number,
        "phone": phone,
        "region": rng.choice(REGIONS),
        "occupation": rng.choice(OCCUPATIONS),
    }
