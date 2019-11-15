# http://www.algo-fx-blog.com/fx-api-oanda-v20-python-trades/
import pandas as pd
import oandapyV20
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.trades as trades
import os, sys
sys.path.append(os.getcwd())
from basic import accountID, access_token, api

# APIへ渡すオーダーのパラメータを作成
data = {
    "order": {
        "instrument": "USD_JPY",
        "units": "+10000",
        "type": "MARKET",
        "positionFill": "DEFAULT"
    }
}

# API経由で買い注文
# r = orders.OrderCreate(accountID, data=data)
# print(api.request(r))

# アカウントのオープントレードを全て取得する
# r = trades.OpenTrades(accountID=accountID)
# print(api.request(r))

'''
- initialMarginRequired : トレードが作成された時点の必要なマージン
- MarginUsed : 現時点でのマージン
- realizedPL : トレードの一部が決済された際の利益/損のトータル
- unrealizedPL : トレードの未決済状態の利益/損のトータル
'''

# TradeID指定して詳細を取得する
# r = trades.TradeDetails(accountID=accountID, tradeID=21)
# print(api.request(r))

# 特定のトレードの一部を決済する
# dataとして決済（クローズ）するパラメータを設定
data = {
    "units": "1000"
}

# TradeID = 21 の1000通過のみ決済する
r = trades.TradeClose(accountID=accountID, tradeID=21, data=data)
print(api.request(r))
