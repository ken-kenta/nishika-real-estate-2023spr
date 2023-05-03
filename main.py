import pandas as pd

from module.preparaion import preparation
from module.training import xgbObj


##############################
# 使用データの準備

train_path = "dataset/all.csv"
test_path = "dataset/test.csv"
train_path_kaggle = "/kaggle/input/nishika-real-estate-2023spr-train-all/all.csv"
test_path_kaggle = "/kaggle/input/nishika-real-estate-2023-spring/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

cols = ['ID', '市区町村コード', '最寄駅：距離（分）', '間取り', '面積（㎡）', '建築年', '建物の構造', '用途',
        '今後の利用目的', '都市計画', '建ぺい率（％）', '容積率（％）', '取引時点', '改装', '取引の事情等']

train_labels = train_df[["ID", "取引価格（総額）_log"]]
train_df = train_df[cols]
test_df = test_df[cols]


##############################
# 前処理

# # （テスト用）数値カラムのみ採用
# num_cols = []
# for col in train_df.columns:
#     if train_df[col].dtype in ("int", "float"):
#         num_cols.append(col)
# num_cols

# train_df = train_df[num_cols]
# test_df = test_df[num_cols]

# カテゴリ変数を数値型に変換
train_df = preparation(train_df)
test_df = preparation(test_df)
# train_dfとtest_dfのカラムをそろえる
diff_cols = list(set(train_df.columns) - set(test_df.columns))
for col in diff_cols:
        if col not in list(train_df.columns):
                train_df[col] = 0
        elif col not in list(test_df.columns):
                test_df[col] = 0
        else:
                 pass


##############################
# 学習・予測

xgbTest = xgbObj(train_df, train_labels)
xgbTest.modeling(num_round=1000)

submission = xgbTest.predict(test_df)
