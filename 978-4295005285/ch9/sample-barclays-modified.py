import requests

def get_mortgages (estimatedPropertyValue, repaymentAmount, months):
    url = 'https://www.barclays.co.uk/dss/service/' + \
        'co.uk/mortgages/costcalculator/productservice'

    headers = {
        # 一般的でないヘッダーがあるのでそれらを含めてしまおう
        'currentState': 'default_current_state',
        'action': 'default',
        'Origin': 'https://www.barclays.co.uk',
        # Referer, User-Agent, X-Requested-With を設定する
        'Referer': 'https://www.barclays.co.uk/mortgages/mortgage-calculator',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64 AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3203.62 Safari/537.36)',
        'X-Requested-With': 'XMLHttpRequest'
    }

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

    r = requests.post(url, json=data, headers=headers)
    results = r.json()
    return results['body']['mortgages']

mortgages = get_mortgages(200000, 150000, 240)
print(mortgages)
