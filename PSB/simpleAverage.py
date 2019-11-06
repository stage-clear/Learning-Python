def sAverage (prices, lookBack, curBar, offset):
    result = 0.0
    for index in range((curBar - offset) - (lookBack - 1), curBar - offset + 1):
        result = result + prices[index]

    result = result / float(lookBack)
    return result
