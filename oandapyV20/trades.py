# trades.TradeDetails(accountID, tradeID) - トレードの詳細を取得する
# trades.TradeClose(accountID, tadeID, *data) - トレードの決済
# trades.TradeCRCDO(accountID, tradeID, *data) - トレードに関する注文を行う
import oandapyV20.endpoints.trades as trades

#---------------------------
# トレードの詳細を取得する
#---------------------------
r = trades.TradeDetails(accountID, tradeID='21')
api.request(r)
r.response

#---------------------------
# トレードの決済
#---------------------------
r = trades.TradeClose(accountID, tradeID='13', data={'units': '10'})
api.request(r)
r.response

#---------------------------
# トレードに関する注文を行う
#---------------------------
r = trades.TradeCRCDO(accountID, tradeID='21', data={
    'stopLoss': {
        'timeInForce': 'GTC',
        'price': '106.870'
    },
    'takeProfit': {
        'timeInForce': 'GTC',
        'price': '106.89'
    }
})
api.request(r)
r.response
