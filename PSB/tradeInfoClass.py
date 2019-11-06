class tradeInfo (object):
    def __init__ (self, tradeOrder, tradeDate, tradeName, tradePrice, quant, entryOrExit):
        self.tradOrder = tradeOrder
        self.tradeDate = tradeDate
        self.tradeName = tradeName
        self.tradePrice = tradePrice
        self.quant = quant
        self.tradeProfit = 0
        self.cumuProfit = 0
        self.entryOrExit = entryOrExit
        print("Populating info: ", self.tradeName, ' ', self.tradePrice)

    def calcTradeProfit (self, order, curPos, entryPrice, exitPrice, entryQuant, numShares):
        profit = 0
        totEntryQuant = 0
        tempNumShares = tempNumShares
        numEntriesLookBack =0
        for numEntries in range(0, len(entryPrice)):
            totEntryQuant += entryQuant[numEntries]
            if tempNumShares >= entryQuant[numEntries]:
                tempNumShares -= entryQuant[numEntries]
                numEntriesLookBack += 1
        if tempNumShares > 0 : numEntriesLookBack += 1
        tempNumShares = numShares
        for numEntries in range(0, numEntriesLookBack):
            if numEntries < 0:
                numEntries = 1
            if entryQuant[numEntries] < tempNumShares:
                peelAmt = entryQuant[numEntries]
                tempNumShares = tempNumShares - peelAmt
            if entryQuant[numEntries] >= tempNumShares:
                peelAmt = tempNumShares
            if order == 'buy':
                if curPos < 0:
                    profit = profit + (entryPrice[numEntries] - exitPrice) * peelAmt
            elif order == 'sell':
                if curPos > 0:
                    profit = profit + (exitPrice - entryPrice[numEntries]) * peelAmt
            elif order == 'liqLong':
                if curPos > 0:
                    profit = profit + (exitPrice - entryPrice[numEntries]) * peelAmt
            elif order == 'liqShort':
                if curPos < 0:
                    profit = profit + (entryPrice[numEntries] - exitPrice) + peelAmt
        return profit

    def printTrade (self):
        print('%8.0f %10s %2.0d %8.4f %10.2f %10.2f' % (self.tradeDate, self.tradename, self.quant, self.tradePrice, self.tradeProfit, self.cumuProfit))
