def highest (prices, lookBack, curBar, offset):
    result = 0.0
    maxVal = 0.00
    for index in range((curBar - offset) - (lookBack - 1), curBar - offset + 1):
        if prices[index] > maxVal:
            maxVal = prices[index]

    result = maxVal
    return result
