class rsiClass (object):
    oldDelta1 = 0

    def __init__ (self):
        self.delta1 = 0
        self.delta2 = 0
        self.rsi = 0
        self.seed = 0

    def calcRsi (self, prices, lookBack, curBar, offset):
        upSum = 0.0
        dnSum = 0.0

        if self.seed == 0:
            self.seed = 1
            for i in range((curBar - offset) - (lookBack - 1), curBar - offset + 1):
                if prices[i] > prices[i - 1]:
                    diff1 = prices[i] - prices[i - 1]
                    upSum += diff1
                if prices[i] < prices[i - 1]:
                    diff2 = prices[i - 1] - prices[i]
                    dnSum += diff2
                self.delta1 = upSum / lookBack
                self.delta2 = dnSum / lookBack
        else:
            if prices[curBar - offset] > prices[curBar - 1 - offset]:
                diff1 = prices[curBar - offset] - prices[curBar - 1 - offset]
                upSum += diff1
            if prices[curBar - offset] < prices[curBar - 1 - offset]:
                diff2 = prices[curBar - 1 - offset] - prices[curBar - offset]
                dnSum += diff2
            self.delta1 = (self.delta1 * (lookBack - 1) + upSum) / lookBack
            self.delta2 = (self.delta2 * (lookBack - 1) + dnSum) / lookBack

        if self.delta1 + self.delta2 != 0:
            self.rsi = (100.0 * self.delta1) / (self.delta1 + self.delta2)
        else:
            self.rsi = 0.0

        return (self.rsi)
