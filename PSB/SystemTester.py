#-----
#Import Section
#form externam modules
#-----

import csv
import tkinter as tk
import os.path
from getData import getData
from dataLists import myData, myTime, myOpen, myHigh, myLow, myClose
from tradeClass import equityClass
from trade import trade
from systemMarket import systemMarketClass
from portfolio import porfolioClass
from indicators import highest, lowest, rsiClass, stochClass, sAverage, bollingerBands
from systemAnalytics import calcSystemResults
from tkinter.fildedialog import askopenfilenames

#-----
#Helper Functions local to this module
#-----
def getDataAtribs (dClass):
    return (dClass.bigPtVal, dClass.symbol, dClass.minMove)

def getDataLists (dClass):
    return (dClass.date, dClass.open, dClass.high, dClass.low, dClass.close)

def calcTodaysOTE (mp, myClose, entryPrice, entryAuant, myBPV):
    todaysOTE = 0
    for entries in range(0, len(entryPrice)):
        if mp >= 1:
            todaysOTE += (myClose - entryPrice[entries])
              * myBPV * entryQuant[entries]
        if mp <= -1:
            todaysOTE += (entryPrice[entries] - myClose)
              * myBPV * entryQuant[entries]

    return (todaysOTE)

def exitPos (myExitPrice, myExitDate, tempName, myCurShares):
    global mp, commission
    global tradeName, entryPrice, entryQuant, exitPrice,
      numShares, myBPV, cumuProfit

    if mp < 0:
        trades = tradeInfo('liqsShort', myExitDate, tempName,
          myExitPrice, myCurShares, 0)
        profit = trades.calcTradeProfit('liqShort', mp, entryPrice, myExitPrice, entryQuant, myCurShares) * myBPV
        profit = profit - myCurShares * commission
        trades.tradeProfit = profit
        cumuProfit += profit
        trades.cumuProfit = cumuProfit
    if mp > 0:
        trades = tradeInfo('liqLong', myExitDate, tempName, myExitPrice, myCurShares, 0)
        profit = trades.calcTradeProfit('liqLong', mp, entryPrice, myExitPrice, entryQuant, myCurShares) * myBPV
        trades.tradeProfit = profit
        profit = profit - myCurShares * commission
        cumuProfit += cumuProfit
        trades.cumuProfit = cumuProfit
    curShares = 0
    for remShares in range(0, len(entryQuant)):
        curShares += entryQuant[remShares]
    return (profit, trades, curShares)

#-----
# End of Functions
#-----

#-----
# Lists and variables are defined and initialized here
#-----
alist, blist, clist, dlist, elist = ([] for i in range(5))
marketPosition, listOfTrades, trueRange, range = ([] for i in range(4))
dataClassList, systemMarketList, equityDataList = ([] for i in range(3))
entryPrice, fileList, entryPrice, entryQuant, exitQuant = ([] for i in range(5))

#exitPrice = list()
currentPrice, totComms, barsSinceEntry = 0
numRuns, myBPV, allowPyra, curShares = 0
#-----
# End of Lists and Variables
#-----

#-----
# Get the raw data and its associates attributes [pointvalue, symbol, tickvalue]
# Read a csv file that has at least D,O,H,L,C - V and OpInt are optional
# Set up a portfolio of multiple markets
#-----

dataClassList = getData()
numMarksets = len(dataClassList)
portfolio = portfolioClass()

def getData ():
    totComms = 0
    with open(filename) as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            commName.append(row[0])
            bigPtVal.append(float(row[1]))
            minMove.append(float(row[2]))
            totComms = totComms + 1
    f.close

#---
root = ti.Tk()
root.withdraw()
cnt = 0
files = askopenfilename(filetypes=(('CSV files', '*.txt'), ('All files', '*.*')), title='Select Markets To Text')

fileList = root.tk.splitlist(files)
filelistLen = len(fileList)

for marksetCnt in range(0, fileListLen):
    head, tail = os.path.split(fileList[marketCnt])
    tempStr = tail[0:2]
    for i in range(totComms):
        if tempStr == commName[i]:
                commIndex = i
    newDataClass = marketDataClass()
    newDataClass.setDataAttribute(commName[commIndex], bigPtVal[commIndex], minMove[commIndex])

    with open(fileList[marksetCnt]) as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            newDataClass.readData(int(row[0]), float(row[1]),
              float(row[2]), float(row[3]), float(row[4]),
              0.0, 0.0)
             cnt = cnt + 1
    dataClassList.append(newDataClass)
    f.close

return dataClassList

class marketDataClass(object):
    def __init__(self):
        self.symbol = ""
        self.minMove = 0
        self.bigPtVal = 0
        self.seed = 0
        self.date = list()
        self.open = list()
        self.high = list()
        self.low = list()
        self.close = list()
        self.volume = list()
        self.opInt = list()
        self.dataPoints = 0
    def setDataAttributes(self, symbol, bigPtVal, minMove):
        self.symbol = symbol
        self.minMove = minMove
        self.bigPtVal = bigPtVal
    def readData (self, date, open, high, low, close, volume, opInt):
        self.date.append(date)
        self.open.append(open)
        self.high.append(high)
        self.low.append(low)
        self.close.append(close)
        self.volume.append(volume)
        self.opInt.append(opInt)
        self.dataPoints += 1

def setEauityInfo(self, equityDate, equityItm, clsTrdEquity, openTrdEquity):
    self.equityDate.append(equityDate)
    self.equityItm.append(equityItm)
    self.cumuClsEquity += clsTrdEquity
    tempEgu = self.cumuClsEauity + open TrdEquity
    self.dailyEquityVal.append(tempEgu)
    maxEgu = self.peakEquity
    self.minEquity = min(self.minEquity, tempEgu)
    self.maxDD = max(self.maxDD, maxEgu - tempEgu)
    maxDD = self.maxDD
    maxDD = maxDD

def createMonthList(li):
    myMonthList = list()

    for i in range(0, len(li)):
        if i != 0:
            tempa = int(li[i] / 100)
            pMonth = int(li[i-1] / 100) % 100
            month = int(li[i] / 100) % 100
            if pMonth != month:
                myMonthList.append(li[i-1])
            if i == len(li) - 1:
                myMonthList.append(li[i])
    return myMonthList

for in in range(0, len(masterDatelist)):
    cumuVal = 0
    for j in range(self.systemMarkets):
        skipDay = 0
        try:
            idx = self.systemMarksets[j].equity.equityDate.index(masterDateList[i])
        except ValueError:
            skipDay = 1

        if skipDay == 0:
            cumuVal += self.systemMarkest[j].equity.dailyEquityVal[idx]
        combineEquity.append(cumuVal)

#-----
# Bollinger Bands
#-----
#Long Entry Logic - bollinger bands
    if  (mp == 0 or mp == -1) and myHigh[i] >= buyLevel:
        profit = 0
        price = max(myOpen[i], buyLevel)
        if mp <= -1:
            profit, trades, curShares = exitPos(price, myDate[i], "RevShrtLiq", curShares)
            listOfTrades.append(trades)
            mp = 0
            todaysCTE = profit
            tradeName = "Boll Buy"
            mp += 1
            marketPosition[i] = mp
            numShares = 1
            entryPrice.append(price)
            entryQuant.append(numShares)
            curShares = curShares + numShares
            trades = tradeInfo('buy', myDate[i], tradeName, entryPrice[-1], numShares, 1)
            barsSinceEntry = 1
            toProfit += profit
            listOfTrades.append(trades)
#Long Exit - Loss
    if mp >= 1 and myLow[i] <= entryPrice[-1] - stopAmt and barsSinceEntry > 1:
        price = min(myOpen[i], entryPrice[-1] - stopAmt)
        tradeName = "L-MMLoss"
        exitDate = myDate[i]
        numShares = curShares
        exitQuant.append(numShares)
        profit, trades, curShares = exitPos(price, myDate[i], tradeName, numShares)
        if curShares == 0 : mp = marketPosition[i] = 0
        totProfit += profit
        todaysCTE = profit
        listOfTrades.append(trades)
        maxPositionL = maxPositionL - 1
# Long Exit - Bollinger Based
    if mp >= 1 and myLow[i] <= exitLevel:
        price = min(myOpen[i], exitLevel)
        tradeName = "L-BollExit"
        numShares = ourShares
        exitQuant.append(numShares)
        profit, trades, curShares = exitPos(price, myDate[i], tradename, numShares)
        if curShares == 0 : mp = marketPosition[i] = 0
        totProfit += profit
        todaysCTE = profit
        listOfTrades.append(trades)
        maxPositionL = maxPositionL - 1

if mp >= 1 and myLow[i] <= exitLevel:
    price = min(myOpen[i], exitLevel)
    tradeName = 'L-BollExit'
    numShares = curShares
    exitQuant.append(numShares)
    profit, trades, curShares = exitPos(price.myDate[i], tradename, numShares)
    if curShares == 0 : mp = marketPosition[i] = 0
    totProfit += profit
    todayCTE = profit
    listOfTrades.append(trades)
    maxPositionL = maxPositionL - 1
    
