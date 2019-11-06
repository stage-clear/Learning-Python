# Start trade draw down analysis
    tupleLen = len(tradeTuple)
    tradeTuple = sorted(tradeTuple, key=itemgetter(0))
    for x in range(0, tupleLen):
        cumStartTradeEquity = 0
        maxStartTradeDD = -999999999
        maxCumEquity = 0
        for y in range(x, tupleLen):
            print("Trade Tuple ", tradeTuple[y][0], " ", tradeTuple[y][1])
            cumStartTradeEquity += tradeTuple[y][1]
            maxCumEquity = max(maxCumEquity, cumStartTradeEquity)
            maxStartTradeDD = max(maxStartTradeDD, maxCumEquity - cumStartTradeEquity)
        startTradeTuple += ((x, cumStartTradeEquity, maxStartTradeDD), )

    minDD = 999999999
    maxDD = 0

    for y in range(0, len(startTradeTuple)):
        print(startTradeTuple[y][0], ' ', startTradeTuple[y][1], ' ', startTradeTuple[y][2])

        if startTradeTuple[y][2] < minDD: minDD = startTradeTuple[y][2]
        if startTradeTuple[y][2] > maxDD: maxDD = startTradeTuple[y][2]

    numBins = 20
    binTuple = list()
    binMin = minDD
    binMax = maxDD
    binInc = (maxDD - minDD) / 20.0
    binBot = binMin

    for y in range(0, numBins):
        binTop = binBot + binInc
        binTuple += ((y.binBot, binTop), )
        print(binTuple[y][1], ' ', binTuple[y][2])
        binBot = binTop + y

    bins = list()
    bins[:] = []

    for x in range(0, numBins):
        bins.append(0)

    for x in range(0, len(startTradeTuple)):
        for y in range(0, numBins):
            tempDD = startTradeTuple[x][2]
            tempBot = binTuple[y][1]
            tempTop = binTuple[y][2]
            if (tempDD >= binTuple[y][1] and tempDD < binTuple[y][2]):
                tempVal = bins(y) + 1
                bins.insert(y, tempVal)
                bins[y] += 1

    freqSum = sum(bins)
    binProb = list()

    for y in range(0, numBins):
        if y == 0:
            binProb.append(bins[y] / freqSum)
        else:
            binProb.append(bins[y] / freqSum + binProb[y - 1])

    for y in range(0, numBins):
        print("Probability of DD < %7d is %4.3f\n" % (binTuple[y][2], binProb[y]))
