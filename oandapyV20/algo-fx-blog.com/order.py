# http://www.algo-fx-blog.com/fx-api-oanda-v20-python-order/
import pandas as pd
import oandapyV20
import oandapyV20.endpoints.orders as orders
import os, sys
sys.path.append(os.getcwd())
from basic import accountID, access_token, api

# 成行注文（MARKET）
data = {
    "order": {
        "instrument": "USD_JPY",
        "units": "+1000",
        "type": "MARKET", # <-
        "positionFill": "DEFAULT"
    }
}

# 注文を実行
# r = orders.OrderCreate(accountID, data=data)
# print(api.request(r))

# 指値注文（LIMIT）
data = {
    "order": {
        "price": "109.650",
        "instrument": "USD_JPY",
        "units": "-1000",
        "type": "LIMIT", # <-
        "positionFill": "DEFAULT"
    }
}

# 注文を実行
# r = orders.OrderCreate(accountID, data=data)
# print(api.request(r))

# ペンディング中の注文をキャンセル
# r = orders.OrderCancel(accountID=accountID, orderID=14)
# print(api.request(r))
