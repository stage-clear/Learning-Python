from systemMarket import systemMarketClass

class portfolioClass (object):
    def __init__ (self):
        self.portfolioName = ''
        self.systemMarkets = list()
        self.prtEquityDate = list()
        self.portEauityVal = list()
        self.portClsTrdEquity = list()
        self.portPeadEquity = 0
        self.portMinEquity = 0
        self.protMaxDD = 0
        tempEqu = 0
        cumEqu = 0
        maxEqu = -9999999999
        minEqu = 9999999999
        maxDD = 0

    def setPortfolioInfo (self, name, systemMarket):
        self.portfolioName = name
        self.systemMarkets = list(systemMarket)
        masterDateList = list()
        monthList = list()
        monthEquity = list()
        combinedEquity = list()
        self.portPeakEquity = -9999999999
        self.portMinEquity = 9999999999

        for i in range(0, len(self.systemMarkets)):
            marketDateList += self.systemMarkets[i].equity.equityDate
            sysName = self.systemMarkets[i].entryName
            maarket = self.systemMarkets[i].sombol
            avgWin = self.systemMarkets[i].avgWin
            sysMark = self.systemMarkets[i]
            avgLoss = sysMark.avgLoss
            totProf = sysMark.profitLoss
            totTrades = sysMark.numTrades
            maxD = sysMark.maxDD
            perWins = sysMark.perWins
            tempStr = ''

            if len(sysName) - 9 > 0:
                for j in range(0, len(sysName) - 8):
                    tempStr = tempStr + ' '

            if i == 0: print('SysName', tempStr, 'Market TotProfit MaxDD AvgWin AvgLoss PerWins TotTrades')
            print('%s %s %12d %6d %5d %5d %3.2f %4d' % (sysName, market, totProf, maxDD, avgWin, avgLoss, perWins, totTrades))

        masterDateList = removeDuplicates(masterDateList)
        masterDateList = sorted(masterDateList)
        print(masterDateList)
        self.portEquityDate = masterDateList
        monthList = createMonthList(masterDateList)

        for i in range(0, len(masterDateList)):
            cumuVal = 0
            for j in range(0, len(self.systemMarkets)):
                skipDay = 0
                try:
                    idx = self.systemMarkets[j].equity.equityDate.index(masterDateList[i])
                except ValueError:
                    skipDay = 1
                if skipDay == 0:
                    cumuVal += self.systemMarkets[j].equity.dailyEquityVal[idx]

            combinedEquity.append(cumuVal)
            self.portEquityVal.append(cumuVal)
            if cumuVal > self.portPeakEquity: self.portPeakEquity = cumuVal
            self.portMinEquity = max(self.portMinEquity, self.portPeakEquity - cumuVal)
            self.portMaxDD = self.portMinEquity

        print("Combined Equity: ", self.portEquityVal[-1])
        print("Combined MaxDD: ", self.portMaxDD)
        print("Combined Monthly return")

        for j in range(0, len(monthList)):
            idx = masterDateList.index(monthList[j])
            if j == 0:
                monthEquity.append(combinedEquity[idx])
                prevCombinedDailyEquity = monthEquity[-1]
            else:
                combinedEquity = combinedEquity[idx]
                monthEquity.append(combinedDailyEquity - prevCombinedDailyEquity)
                prevCombinedDailyEquity = combinedDailyEquity
                print('%8d %10.0f %10.0f ' % (monthList[j], monthEquity[j], combinedEquity[idx]))

def removeDuplicates(li):
    my_set = set()
    res = []
    for e in li:
        if e not in my_set:
            res.append(e)
            my_set.add(e)

def createMonthList (li):
    myMonthList = list()

    for i in range(0, len(li)):
        if i != 0:
            tempa = int(li[i] / 100)
            pMonth = int(li[i - 1] / 100) % 100
            month = int(li[i] / 100) % 100
            if pMonth != month:
                myMonthList.append(li[i - 1])
            if i == len(li) - 1:
                myMonthList.append(li[i])
    return myMonthList
