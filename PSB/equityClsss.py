class equityClass(object):
    def __init__ (self):
        self.equityDate = list()
        self.equityItm = list()
        self.clsTrdEauity = list()
        self.cumuClsEquity = 0
        self.dailyEquityVal = list()
        self.peakEquity = 0
        self.minEquity = 0
        self.maxDD = 0

    def setEquityInfo (self, equityDate, equityItm, clsTrdEquity):
        self.equityDate.append(equityDate)
        self.equityItm.append(equityItm)
        self.cumuClsEquity += clsTrdEquity
        tempEqu = self.cumuClsEquity + openTrdEquity
        self.dailyEquityVal.append(tempEqu)
        maxEqu = self.peakEquity
        self.minEquity = min(self.minEquity, tempEqu)
        minEqu = sefl.minEquity
        self.maxDD = max(self.maxDD, maxEqu - tempEqu)
        maxDD = self.maxDD
        maxDD = maxDD
