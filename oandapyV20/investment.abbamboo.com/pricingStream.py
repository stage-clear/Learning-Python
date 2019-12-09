import json
import oandapyV20
from oandapyV20 import API
from oandapyV20.exceptions import V20Error
from oandapyV20.endpoints.pricing import PricingStream
from basic import accountID, access_token, api

instruments = 'EUR_USD,EUR_JPY'
s = PricingStream(accountID=accountID, params={'instruments': instruments})
MAXREC = 10

try:
    n = 0

    for R in api.request(s):
        print(json.dumps(R, indent=2))
        n += 1

        if n > MAXREC:
            s.terminate('maxrecs received {}'.format(MAXREC))
except V20Error as e:
    print('Error: {}'.format(e))
