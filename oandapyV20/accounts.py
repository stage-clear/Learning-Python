# accounts.AccountSummary(accountID) - アカウントの残高などを取得する
import oandapyV20.endpoints.accounts as accounts

#---------------------------
# アカウントの残高などを取得する
#---------------------------
r = accounts.AccountSummary(accountID)
api.request(r)
r.response
