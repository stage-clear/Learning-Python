def sAverage (prices, lookback, curBar, offset):
    result = 0.0
    for index in range((curBar - offset) - (lookback - 1), curBar - offset + 1):
        result = result + prices[index]
    result = result / float(lookback)
    return result

def highest (prices, lookback, curBar, offset):
    result = 0.0
    maxVal = 0.00
    for index in range((curBar - offset) - (lookback - 1), curBar - offset + 1):
        if prices[index] > maxVal:
            maxVal = prices[index]
    result = maxVal
    return result

def lowest (prices, lookback, curBar, offset):
    result = 0.0
    minVal = 999999.0
    for index in range((curBar - offset) - (lookback - 1), curBar - offset + 1):
        if prices[index] < minVal:
            minVal = prices[index]
    result = minVal
    return result

class rsiClass(object):
    oldDelta1 = 0
    def __init__ (self):
        self.delta1 = 0
        self.delta2 = 0
        self.rsi = 0
        self.seed = 0
    def calcRsi (self, prices, lookback, curBar, offset):
        upSum = 0.0
        dnSum = 0.0
        if self.seed == 0:
            self.seed = 1
            for i in range((curBar - offset) - (lookback - 1), curBar - offset):
                if prices[i] > prices[i - 1]:
                    diff1 = prices[i] - prices[i - 1]
                    upSum += diff1
                if prices[i] < prices[i - 1]:
                    diff2 = prices[i - 1] - prices[i]
                    dnSum += diff2
                self.delta1 = upSum / lookback
                self.delta2 = dnSum / lookback
        else:
            if prices[curBar - offset] > prices[curBar - 1 - offset]:
                diff1 = prices[curBar - offset] - prices[curBar - 1 - offset]
                upSum += diff1
            if prices[curBar - offset] < prices[curBar - 1 - offset]:
                diff2 = prices[curBar - 1 - offset]
                dnSum += diff2
            self.delta1 = (self.delta1 * (lookback - 1) + upSum) / lookback
            self.delta2 = (self.delta2 * (lookback - 1) + dnSum) / lookback

        if self.delta1 + self.delta2 != 0:
            self.rsi = (100.0 * self.delta1) / (self.delta1 + self.delta2)
        else:
            self.rsi = 0.0

        return (self.rsi)
