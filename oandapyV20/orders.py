# orders.OrderCreate(accountID, data) - オーダーを作成する
# orders.OrdersPending(accountID, orderID) - 保留中のオーダー情報を取得する
# orders.OrderDetails(accountID, orderID) - オーダーの詳細を取得する
# orders.OrderCancel(accountID, orderID) - オーダーをキャンセルする
import oandapyV20.endpoints.orders as orders

#---------------------------
# 成り行き注文でエントリー
#---------------------------
order_data = {
    'order': {
        'instrument': 'USD_JPY',
        'units': '+100',
        'type': 'MARKET'
    }
}

o = orders.OrderCreate(accountID, data=order_data)
api.request(o)
o.response

#---------------------------
# 逆指値注文でエントリー
#---------------------------
order_data = {
    'order': {
        'price': '107.765',
        'instrument': 'USD_JPY',
        'units': '+10000',
        'type': 'STOP'
    }
}

o = orders.orderCreate(accountID, data=order_data)
api.request(o)
o.response

#---------------------------
# 保留中のオーダー情報を取得する
#---------------------------
c = orders.OrderPending(accountID)
api.request(c)
c.response

#---------------------------
# オーダーの詳細を取得する
#---------------------------
c = orders.OrderDetails(accountID, orderID='133')
api.request(c)
c.response

#---------------------------
# オーダーをキャンセルする
#---------------------------
c = orders.OrderCancel(accountID, orderID='18')
api.request(c)
c.response
