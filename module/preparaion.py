import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


def get_num(text):
    try:
        return int(re.findall('\d+', text)[0])
    except Exception:
        return None


def era_name_converter(text):
    if text == '昭和':
        return 1925
    elif text == '平成':
        return 1988
    elif text == '令和':
        return 2018
    else:
        return None


def time_to_station(arg):
    try:
        text = arg[0]
    except Exception:
        text = arg

    nums = list(re.findall('\d+', str(text)))
    if len(nums) == 1:
        num = int(nums[0])
    elif len(nums) == 2:
        num = int(nums[0]) * 60 + int(nums[1])
    else:
        num = None

    return num


def preparation(data):
    # 改装
    data['改装済'] = 0
    data.loc[data['改装'] == '改装済', '改装済'] = 1
    data = data.drop(columns='改装')

    # 取引の事情等
    data['事情有'] = 0
    data.loc[data['取引の事情等'] == True, '事情有'] = 1
    data = data.drop(columns='取引の事情等')

    # 面積（㎡）
    data["面積（㎡）"] = data["面積（㎡）"].apply(get_num).astype(float)

    # 建築年、取引時点
    data['建築年（年号）'] = data['建築年'].str[:2]
    data['建築年（年号）'] = data['建築年（年号）'].apply(era_name_converter)
    data['建築年'] = data['建築年'].apply(get_num)
    data['取引時点'] = data['取引時点'].str[:4].astype(int)
    data['築年数'] = data['取引時点'] - data['建築年（年号）'] - data['建築年']
    data.loc[data['築年数'] < 0, '築年数'] = None
    data = data.drop(columns=['建築年（年号）', '建築年', '取引時点'])

    # 間取り
    data['L'] = data['間取り'].str.count('Ｌ')
    data['D'] = data['間取り'].str.count('Ｄ')
    data['K'] = data['間取り'].str.count('Ｋ')
    data['LDK以外'] = data['間取り'].apply(get_num)
    data['部屋数'] = data['L'] + data['D'] + data['K'] + data['LDK以外']
    data = data.drop(columns='間取り')

    # 最寄駅：距離（分）
    data['最寄駅'] = data['最寄駅：距離（分）'].str.split('?')
    data['最寄駅'] = data['最寄駅'].apply(time_to_station)
    data = data.drop(columns='最寄駅：距離（分）')

    # 今後の利用目的
    data = pd.get_dummies(data, columns=['今後の利用目的'])

    # 用途
    youto = data['用途'].str.split('、', expand=True).fillna("null")
    for x in list(set(youto[0].unique()) | set(youto[1].unique())):
        youto.loc[(youto[0] == x) | (youto[1] == x), f"用途_{x}"] = 1
    youto = youto.drop(columns="用途_null")
    data = data.merge(youto.drop(columns=[0, 1]), left_index=True, right_index=True)
    data = data.drop(columns='用途')

    # 建物の構造
    kouzou = data['建物の構造'].str.split('、', expand=True) \
                                    .fillna('null') \
                                    .replace({'鉄骨造': 'S', 'ブロック造': 'B', '木造': 'W', '軽量鉄骨造': 'S', 'null': None})
    kouzou.loc[(kouzou[0] == 'ＳＲＣ') | (kouzou[0] == 'ＳＲＣ') | (kouzou[1] == 'ＳＲＣ'), 'SRC'] = 1
    kouzou.loc[(kouzou[0] == 'ＲＣ') | (kouzou[0] == 'ＲＣ') | (kouzou[1] == 'ＲＣ'), 'RC'] = 1
    kouzou.loc[(kouzou[0] == 'S') | (kouzou[0] == 'S') | (kouzou[1] == 'S'), 'S'] = 1
    kouzou.loc[(kouzou[0] == 'B') | (kouzou[0] == 'B') | (kouzou[1] == 'B'), 'B'] = 1
    kouzou.loc[(kouzou[0] == 'W') | (kouzou[0] == 'W') | (kouzou[1] == 'W'), 'W'] = 1
    data = data.merge(kouzou.drop(columns=[0, 1]), left_index=True, right_index=True)
    data = data.drop(columns='建物の構造')

    # 都市計画
    data.loc[data['都市計画'].str.contains('住居') == True, '住居系地域'] = 1
    data.loc[data['都市計画'].str.contains('工業') == True, '工業系地域'] = 1
    data.loc[data['都市計画'].str.contains('商業') == True, '商業系地域'] = 1
    data = data.drop(columns='都市計画')
    
    return data
