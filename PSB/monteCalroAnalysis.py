# Monte Carlo Analysis

    mcTradeTuple = list()

    for x in range(0, 5): # number of alternate histories
        for y in range(0, len(tradeTuple)):
            randomTradeNum = randam.randint(0, len(tradeTuple) - 1)
            mcTradeTuple += ((x, y, tradeTuple[randomTradeNum][1], tradeTuple[randomTradeNum][0]), )

    mcTradeResultTuple = list()
    whichAlt = -1

    for x in range(0, len(mcTradeTuple)):
        if mcTradeTuple[x][1] == 0:
            print('New Alternate History Generated')
            cumEquity = 0
            maxTradeDD = -99999
            maxCumEquity = 0

        cumEquity += mcTradeTuple[x][2]
        print('Randomized trade listing : ', mcTradeTuple[x][3], ' ', mcTradeTuple[x][2])
        maxCumEquity = max(maxCumEquity, cumEquity)
        maxTradeDD = max(maxTradeDD, maxCumEquity - cumEquity)

        if mcTradeTuple[x][1] == len(tradeTuple) - 1:
            mcTradeResultsTuple += ((cumEquity, maxTradeDD, cumEquity / len(tradeTuple)), )

    for x in range(0, len(mcTradeResultsTuple)):
        mcCumEquity = mcTradeResultsTuple[x][0]
        mcMaxDD = mcTradeResultsTuple[x][1]
        mcAvgTrd = mcTradeResultsTuple[x][2]
        print('Alt history %5d Profit: %10d MacDD: %10d Avg Trade %6d\n' % (x, mcCumEquity, mcMacDD, mcAvgTrd))
