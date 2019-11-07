import requests

url = 'https://www.barclays.co.uk/dss/service/co.uk/mortgages/' + \
    'costcalculator/productservice'

session = requests.Session()

estimatedPropertyValue = 200000
repaymentAmount = 150000
months = 240
data = {
    'header': { 'flowId': '0'},
    'body': {
        'wantTo': 'FTBP',
        'estimatedPropertyValue': estimatedPropertyValue,
        'borrowAmount': repaymentAmount,
        'interestOnlyAmount': 0,
        'repaymentAmount': repaymentAmount,
        'ltv': round(repaymentAmount / estimatedPropertyValue * 100),
        'totalTerm': months,
        'purchaseType': 'Repayment'
    }
}

r = session.post(url, json=data)

print(r.json())
