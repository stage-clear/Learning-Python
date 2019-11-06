def bollingerBands (dates, prices, lookBack, numDevs, curBar, offset):
    sum1 = 0.0
    sum2 = 0.0
    startPt = (curBar - offset) - (lookBack - 1)
    endPt = curBar - offset + 1

    for index in range(startPt, endPt):
        tempDate = dates[index]
        sum1 = sum1 + prices[index]
        sum2 = sum2 + prices[index] ** 2

    mean = sum1 / float(lookBack)

    stdDev = ((lookBack * sum2 - sum1 ** 2) / (lookBack * (lookBack - 1))) ** 0.5

    upBand = mean + numDevs * stdDev
    dnBand = mean - numDevs * stdDev

    print(mean, " ", stdDev, " ", upBand, " ", dnBand)
    return upBand, dnBand, mean
