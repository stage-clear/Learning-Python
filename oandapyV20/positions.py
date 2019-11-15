# positions.PositionClose(accountID) - ポジションを決済する
# positions.PositionDetails(accountID) - ポジションの詳細を取得する
import oandapyV20.endpoints.positions as positions

#---------------------------
# ポジションを決済する
#---------------------------
position_data = { 'longUnits': 'ALL'}
# position_data = { 'longUnits': '1000' }

p = positions.PositionClose(accountID=accountID, data=position_data, instrument='USD_JPY')
api.request(p)
p.response

#---------------------------
# ポジションの詳細を取得する
#---------------------------
p = positions.PositionDetails(accountID=accountID, instrument='USD_JPY')
api.request(p)
p.response
