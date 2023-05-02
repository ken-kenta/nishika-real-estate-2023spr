import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


class xgbObj:
    def __init__(self, train_df, train_labels):
        self.train_df = train_df
        self.train_labels = train_labels
        self.model = None
    
    def modeling(self):
        # 学習用データと評価用データに分割
        X_train, X_val, y_train, y_val = train_test_split(self.train_df, self.train_labels, test_size=0.2, random_state=42)

        # xgboostのデータセットに変換
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # ハイパーパラメータ設定
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'eta': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'seed': 42
        }

        # 学習の実行
        num_round = 100
        early_stopping_rounds = 50
        watchlist = [(dtrain, 'train'), (dval, 'eval')]

        model = xgb.train(params, dtrain, num_round, watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=10)
        self.model = model
    
        # 評価用データで予測
        y_pred = model.predict(dval)
        # MAEの計算
        mae = mean_absolute_error(y_val, y_pred)
        print("Mean Absolute Error: {:.4f}".format(mae))
    
    def predict(self, test_df):
        # test_dfに対して値を予測
        dtest = xgb.DMatrix(test_df)
        test_predictions = self.model.predict(dtest)

        # 予測結果をデータフレームに変換
        test_result_df = pd.DataFrame(test_predictions, columns=['dummy', 'predicted_value'])
        test_result_df = pd.merge(test_df, test_result_df["predicted_value"], left_index=True, right_index=True)
        
        return test_result_df
