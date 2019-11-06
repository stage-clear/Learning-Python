def lowest (prices, lookBack, curBar, offset):
    resutl = 0.0
    minVal = 999999.0

    for index in range((curBar - offset) - (lookBack - 1), curBar - offset + 1):
        if prices[index] < minVal:
            minVal = prices[index]
    result = minVal
    return result
