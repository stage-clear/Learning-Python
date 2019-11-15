# http://www.algo-fx-blog.com/fx-api-oanda-v20-python-positions-management/
import pandas as pd
import oandapyV20
import oandapyV20.endpoints.positions as positions
import os, sys
sys.path.append(os.getcwd())
from basic import accountID, access_token, api

# 講座のすべてのポジションをリストとして取得
r = positions.PositionList(accountID=accountID)
print(api.request(r))

# 通貨を指定してポジションを取得
r = positions.PositionDetails(accountID=accountID, instrument="EUR_GBP")
print(api.request(r))
# ポジションを保有していない場合はエラー

# すべてのポジションを決済する
data = {
    "longUnits": "ALL"
}

# ドル円の買い注文全ての保有ポジションを決済する
r = positions.PositionClose(accountID=accountID, data=data, instrument="USD_JPY")
print(api.request(r))
