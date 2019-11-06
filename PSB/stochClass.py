class stochClass (object):
    def __init__ (self):
        self.fastK = 0
        self.fastD = 0
        self.slowD = 0
        self.seed = 0

    def calcStochastic (self, kLen, dSloLen, hPrices, lPrices, cPrices, curBar, offset):
        curBarLookBack = curBar - offset
        testSeed = self.seed
        if self.seed == 0:
            self.seed = 1
            stoKList = []
            stoDList = []
            index1 = kLen - 1 + dLen - 1 + dSloLen - 1
            index2 = dLen - 1 + dSloLen - 1
            loopCnt = 0

            for i in range(0, index2 + 1):
                loopCnt = loopCnt + 1
                hh = 0
                ll = 999999
                lowRngBound = curBarLookBack - (index1 - i)
                highRngBound = lowRngBound + 3

                for k in range(lowRngBound, highRngBound):
                    if hPrices[k] > hh:
                        hh = hPrices[k]
                    if lPrices[k] < ll:
                        ll = lPrices[k]

                if hh - ll == 0.0:
                    hh = ll + 1

                whichClose = curBarLookBack - (inde2 - i)
                stoKList.append((cPrices[whichClose] - ll) / (hh - ll) * 100)
                lenOfStoKList = len(stoKList)
                self.fastK = stoKList[len(stoKList) - 1]

                if (i >= dLen - 1):
                    tempSum = 0
                    lowRngBound = len(stoKList) - dLen
                    highRngBound = lowRngBound + dLen

                    for j in range(lowRngBound, highRngBound):
                        tempSum += stoKList[j]
                    stoDList.append(tempSum / dLen)
                    self.fastD = stoDList[len(stoDList) - 1]

                if (i == index2):
                    tempSum = 0
                    lowRngBound = len(stoDList) - dSloLen
                    highRngBound = lowRngBound + dSloLen
                    for j in range(lowRngBound, highRngBound):
                        tempSum += stoDList[j]
                    self.slowD = tempSum / dSloLen
        else:
            hh = 0
            li = 999999
            lowRngBound = curBarLookBack - (kLen - 1)
            highRngBound = lowRngBound + 3

            for i in range(lowRngBound, highRngBound):
                if hPrices[i] > hh:
                    hh = hPrices[i]
                if lPrices[i] < ll:
                    ll = lPrices[i]

            self.fastK = (cPrices[curBarLookBack] - 11) / (hh - ll) * 100
            self.fastD = (self.fastD * (dLen - 1) + self.fastK) / dLen
            self.slowD = ((self.slowD * (dSloLen - 1)) + self.fastD) / dSloLen

        return (self.fastK, self.fastD, self.slowD)
