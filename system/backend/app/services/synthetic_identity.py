from __future__ import annotations

import hashlib
import random


FAMILY_NAMES = ["Li", "Wang", "Zhang", "Liu", "Chen", "Yang", "Huang", "Zhao", "Wu", "Zhou"]
GIVEN_NAMES = ["Wei", "Fang", "Min", "Jie", "Tao", "Lei", "Yan", "Qiang", "Xuan", "Ning"]
REGIONS = ["Shanghai", "Beijing", "Hangzhou", "Shenzhen", "Chengdu", "Nanjing", "Wuhan", "Suzhou"]
OCCUPATIONS = ["merchant", "student", "driver", "engineer", "accountant", "sales", "freelancer"]


def synthetic_person_for_node(node_id: str) -> dict[str, str]:
    seed = int(hashlib.sha256(str(node_id).encode("utf-8")).hexdigest()[:12], 16)
    rng = random.Random(seed)
    display_name = f"{rng.choice(FAMILY_NAMES)} {rng.choice(GIVEN_NAMES)}"
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
