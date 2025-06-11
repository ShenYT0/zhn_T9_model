import asyncio

async def pinyin_to_hanzi(pinyin: str):
    return ["你", "好", "是", "在", "他", "啦", "的", "中", "和", "国", "人", "们", "来", "了", "就", "说", "要", "不", "有"]

from itertools import product

async def code_to_pinyin(code: str):
    key_labels = {
        "2": "abc",
        "3": "def",
        "4": "ghi",
        "5": "jkl",
        "6": "mno",
        "7": "pqrs",
        "8": "tuv",
        "9": "wxyz"
    }

    letters_list = []
    for digit in code:
        letters_list.append(key_labels[digit])

    combinations = product(*letters_list)

    return [''.join(item) for item in combinations]

async def code_to_hanzi(code: str):
    return ["泥", "你", "尼", "呢", "内", "那", "哪", "娜", "纳", "南", "难", "男", "脑", "闹", "挠", "闹"]

async def predict_next_hanzi(current_input: str):
    return ["是", "在", "的", "来", "了", "就", "说", "要", "不", "有"]