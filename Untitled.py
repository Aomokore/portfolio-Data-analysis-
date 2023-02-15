#!/usr/bin/env python
# coding: utf-8

# In[661]:


a=-5
b=6
c=a+b
print(c)


# In[662]:


abs(a)


# In[663]:


import yfinance as yf
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 


# In[664]:


Apple = yf.download("AAPL", start = "2010-01-01", end= "2021-01-01")


# In[665]:


Apple


# In[666]:


a= [1,2,3,4]
a


# In[667]:


tickers=["SPY", "AAPL","KO"]


# In[668]:


Stocks = yf.download(tickers, start = "2010-01-01", end= "2021-01-01")


# In[669]:


Stocks


# In[670]:


Stocks.head()


# In[671]:


Stocks.tail()


# In[672]:


Stocks.info()


# In[673]:


Stocks.to_csv("stocksYT.csv")


# In[674]:


stocks = pd.read_csv("stocksYT.csv")


# In[675]:


stocks


# In[676]:


stocks = pd.read_csv("stocksYT.csv", header=[0,1], index_col=[0])


# In[677]:


stocks


# In[678]:


stocks = pd.read_csv("stocksYT.csv", header=[0,1], index_col=[0], parse_dates=[0])


# In[679]:


stocks


# In[680]:


stocks.columns


# In[681]:


stocks.columns= stocks.columns.to_flat_index()


# In[682]:


stocks.columns


# In[683]:


stocks


# In[684]:


stocks.columns=pd.MultiIndex.from_tuples(stocks.columns)


# In[685]:


stocks


# In[686]:


stocks.columns


# In[687]:


stocks.describe()


# In[688]:


close= stocks.loc[:,"Close"].copy()


# In[689]:


close


# In[690]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 
plt.style.use("seaborn")


# In[691]:


close.plot(figsize=(15,8),fontsize=12)
plt.legend(fontsize=12)
plt.show()


# In[692]:


close.iloc[0,0]


# In[693]:


close.AAPL.div(close.iloc[0,0])


# In[694]:


close.AAPL.div(close.iloc[0,0]).mul(100)


# In[695]:


close.div(close.iloc[0]).mul(100)


# In[696]:


normclose= close.div(close.iloc[0]).mul(100)


# In[697]:


normclose.plot(figsize=(15,8),fontsize=12)
plt.legend(fontsize=12)
plt.show()


# In[698]:


aapl= close.AAPL.copy().to_frame()


# In[699]:


aapl


# In[700]:


aapl.shift(periods=1)


# In[701]:


aapl["lag1"]=aapl.shift(periods=1)


# In[702]:


aapl


# In[703]:


aapl.AAPL.sub(aapl.lag1)


# In[704]:


aapl["diff"]=aapl.AAPL.sub(aapl.lag1)


# In[705]:


aapl


# In[706]:


aapl.AAPL.div(aapl.lag1)


# In[707]:


aapl["%change"]=aapl.AAPL.div(aapl.lag1).sub(1).mul(100)


# In[708]:


aapl


# In[709]:


aapl["Diff2"]= aapl.AAPL.diff(periods=1)


# In[710]:


aapl


# In[711]:


aapl["%change 2"]=aapl.AAPL.pct_change(periods=1).mul(100)


# In[712]:


aapl


# In[758]:


del aapl["change"]


# In[718]:


aapl


# In[719]:


aapl.rename(columns= {'%change 2':'change'}, inplace= True)


# In[720]:


aapl


# In[721]:


aapl.AAPL.resample("M").last()


# In[722]:


aapl.AAPL.resample("BM").last().pct_change(periods=1).mul(100)


# In[723]:


ret= aapl.pct_change().dropna()


# In[724]:


ret.info()


# In[725]:


aapl.AAPL.resample("BM").last().pct_change(periods=1).mul(100)


# In[726]:


ret.info()


# In[727]:


aapl


# In[728]:


aapl.AAPL.resample("BM").last().pct_change(periods=1).mul(100)


# In[761]:


ret= aapl.pct_change().dropna()


# In[762]:


ret.info()


# In[763]:


ret.plot(kind="hist",figsize=(12,8),bins=100)
plt.show()


# In[764]:


daily_mean_ret= ret.mean()
daily_mean_ret


# In[765]:


var_daily = ret.var()
var_daily


# In[766]:


std_daily= np.sqrt(var_daily)


# In[767]:


std_daily


# In[768]:


annual_mean_ret= daily_mean_ret*252
annual_mean_ret


# In[769]:


annual_var_ret= var_daily*252
annual_var_ret


# In[770]:


annual_std_ret= np.sqrt(annual_var_ret)


# In[771]:


annual_std_ret


# In[772]:


ret.std()*np.sqrt(252)


# In[773]:


tickers=["SPY", "AAPL","KO","IBM","DIS","MSFT"]


# In[774]:


Stocks = yf.download(tickers, start = "2010-01-01", end= "2021-01-01")


# In[775]:


close= Stocks.loc[:,"Close"].copy()


# In[776]:


normclose= close.div(close.iloc[0]).mul(100)


# In[777]:


normclose.plot(figsize=(15,8),fontsize=12)
plt.legend(fontsize=12)
plt.show()


# In[778]:


ret=close.pct_change().dropna()
ret.head()


# In[779]:


ret.describe().T


# In[780]:


summary=ret.describe().T.loc[:,["mean", "std"]]
summary


# In[781]:


summary["mean"]= summary["mean"]*252
summary["std"]=  summary["std"]*np.sqrt(252)
summary


# In[782]:


summary.plot.scatter(x="std",y="mean",figsize=(12,8),s=50,fontsize=15)
for i in summary.index:
    plt.annotate(i,xy=(summary.loc[i,"std"]+0.002,summary.loc[i,"mean"]+0.002),size=15)
plt.xlabel("annual risk(std)", fontsize=15)
plt.ylabel("annual return", fontsize=15)
plt.title("risk/return", fontsize=25)
    


# In[783]:


fruits= ["aaple", "banana", "mango"]
for i in fruits:
    print(i)


# In[784]:


fruits[1]


# ##correlation and covariance
# 

# In[785]:


ret


# In[786]:


ret.cov()


# In[787]:


ret.corr()


# In[788]:


import seaborn as sns


# In[789]:


plt.figure(figsize=(12,8))
sns.set(font_scale=1.4)
sns.heatmap(ret.corr(),cmap="Reds",annot=True, annot_kws={"size":15},vmax=0.6)
plt.show()


# In[790]:


tickers=["SPY", "AAPL","KO","IBM","DIS","MSFT","META","MGI","GOOGL","AZN","GSK","PFE"]


# In[791]:


Stocks = yf.download(tickers, start = "2010-01-01", end= "2021-01-01")


# In[792]:


close= Stocks.loc[:,"Close"].copy()


# In[793]:


normclose= close.div(close.iloc[0]).mul(100)


# In[794]:


normclose.plot(figsize=(15,8),fontsize=12)
plt.legend(fontsize=12)
plt.show()


# In[795]:


ret=close.pct_change().dropna()
ret.head()


# In[796]:


ret.describe().T


# In[797]:


summary=ret.describe().T.loc[:,["mean", "std"]]
summary


# In[798]:


summary["mean"]= summary["mean"]*252
summary["std"]=  summary["std"]*np.sqrt(252)
summary


# In[799]:


summary.plot.scatter(x="std",y="mean",figsize=(12,8),s=50,fontsize=15)
for i in summary.index:
    plt.annotate(i,xy=(summary.loc[i,"std"]+0.002,summary.loc[i,"mean"]+0.002),size=15)
plt.xlabel("annual risk(std)", fontsize=15)
plt.ylabel("annual return", fontsize=15)
plt.title("risk/return", fontsize=25)


# In[800]:


ret.corr()


# In[801]:


ret.cov()


# In[802]:


plt.figure(figsize=(12,8))
sns.set(font_scale=1.4)
sns.heatmap(ret.corr(),cmap="Reds",annot=True, annot_kws={"size":15},vmax=0.6)
plt.show()


# simple returns and log returns

# In[803]:


df= pd.DataFrame(index=[2016,2017,2018],data=[100,50,95],columns=["price"])


# In[804]:


df


# In[805]:


simplereturns=df.pct_change().dropna()
simplereturns


# In[806]:


simplereturns.mean()


# In[807]:


#meanreturnsaremisleading


# In[808]:


logreturns=np.log(df/df.shift(1)).dropna()
logreturns


# In[809]:


logreturns.mean()


# In[810]:


100*np.exp(logreturns.mean()*2)


# In[811]:


SPY = yf.download("SPY")


# In[812]:


spy=SPY.Close.to_frame()


# In[813]:


spy


# In[814]:


spy.plot(figsize=(12,8),fontsize=15)
plt.legend(loc="upper left", fontsize=15)
plt.show()


# In[815]:


spy_roll=spy.rolling(window=10)


# In[816]:


spy_roll=spy_roll.mean()


# In[817]:


spy_roll.head(15)


# In[818]:


spy.rolling(window=10).max()


# In[819]:


spy_roll=spy.rolling(window=10)


# In[820]:


spy.rolling(window=10, min_periods=5).max()


# In[821]:


spy


# In[822]:


spy["SMA"]= spy.rolling(window=50, min_periods=50).mean()


# In[823]:


spy


# In[824]:


spy.plot(figsize=(12,8),fontsize=15)
plt.legend(loc="upper left", fontsize=15)
plt.show()


# In[825]:


spy["SMA200"]= spy.Close.rolling(window=200, min_periods=200).mean()
spy


# In[826]:


spy.plot(figsize=(12,8),fontsize=15)
plt.legend(loc="upper left", fontsize=15)
plt.show()


# In[827]:


spy["EMA200"]= spy.Close.ewm(span=100,min_periods=100).mean()


# In[828]:


spy


# In[829]:


spy.plot(figsize=(12,8),fontsize=15)
plt.legend(loc="upper left", fontsize=15)
plt.show()


# In[830]:


spy["Day"]=spy.index.day_name()


# In[831]:


spy


# In[832]:


spy["Quarter"]=spy.index.quarter


# In[833]:


spy


# In[834]:


spy= yf.download("SPY")


# In[835]:


del spy["Quarter"]


# In[836]:


spy


# In[837]:


spy=spy.Close.to_frame()


# In[838]:


spy


# In[839]:


all_days=pd.date_range(start="2010-12-31", end="2020-12-31", freq="D")
all_days


# In[840]:


spy


# In[841]:


spy=spy.reindex(all_days)


# In[842]:


spy


# In[843]:


spy.fillna(method="ffill")


# In[844]:


spy.fillna(method="bfill")


# In[845]:


SPY = yf.download("SPY", interval= "1wk")


# In[846]:


SPY


# ##cummaltive returns& drawdowns 

# In[847]:


apple= yf.download("AAPL")


# In[848]:


apple=apple.Close.to_frame()


# In[849]:


apple


# In[850]:


apple["d_returns"]=np.log(apple.div(apple.shift(1)))


# In[851]:


apple


# In[852]:


apple.dropna(inplace= True)


# In[853]:


apple


# In[854]:


apple.d_returns.sum()


# In[855]:


np.exp(apple.d_returns.sum())


# In[856]:


apple["cummreturns"]= apple.d_returns.cumsum().apply(np.exp)


# In[857]:


apple


# In[858]:


apple.cummreturns.plot(figsize=(12,8),title="APPLE buy and hold", fontsize=12)
plt.show()


# In[859]:


apple.d_returns.mean()*252


# In[860]:


apple.d_returns.std()*np.sqrt(252)


# ##markdown

# In[861]:


apple["cummax"]=apple.cummreturns.cummax()


# In[862]:


apple


# In[863]:


apple[["cummreturns","cummax"]].plot(figsize=(12,8),title="APPLE buy and hold+ cummax", fontsize=12)
plt.show()


# In[864]:


apple["drawdown"]=apple["cummax"]-apple["cummreturns"]


# In[865]:


apple


# In[866]:


apple.drawdown.max()


# In[867]:


apple.drawdown.idxmax()


# In[868]:


apple.loc[(apple.index<='2023-01-05')]


# In[869]:


apple["drawdown%"]= (apple["cummax"]-apple["cummreturns"])/apple["cummax"]


# In[870]:


del apple["%drawdown"]


# In[871]:


apple


# In[872]:


apple["drawdown%"].max()


# In[873]:


apple["drawdown%"].idxmax()


# In[874]:


apple.loc[(apple.index<='1997-12-23')]


# ##sma Strategy

# In[900]:


data=apple.Close.loc[(apple.index<='2023-01-05')].to_frame()


# In[901]:


data


# In[902]:


sma_s=50
sma_l=100


# In[903]:


data["sma_s"]=data.Close.rolling(sma_s).mean()
data["sma_l"]=data.Close.rolling(sma_l).mean()


# In[904]:


data


# In[905]:


data.plot(figsize=(12,8), title="AAPLE- SMA{} | SMA{}".format(sma_s,sma_l),fontsize=12)


# In[922]:


data.loc["2016"].plot(figsize=(12,8), title="AAPLE- SMA{} | SMA{}" .format(sma_s,sma_l),fontsize=12)


# In[923]:


data.dropna(inplace= True)


# In[924]:


data


# In[925]:


data["position"]= np.where(data["sma_s"]>data["sma_l"],1,-1)


# In[926]:


data


# In[927]:


data.loc[:, ["sma_s","sma_l","position"]].plot(figsize=(12,8), title="AAPLE- SMA{} | SMA{}".format(sma_s,sma_l),fontsize=12)


# In[938]:


data.loc["2016", ["sma_s","sma_l","position"]].plot(figsize=(12,8), title="AAPLE- SMA{} | SMA{}".format(sma_s,sma_l),fontsize=12, secondary_y="position")


# In[1037]:


data["returnb&h"]=np.log(data.Close.div(data.Close.shift(1)))


# In[1038]:


data


# In[1053]:


data["strategy"]= data["returnb&h"] * data.position.shift(1)


# In[1054]:


data


# In[1055]:


data.dropna(inplace= True)


# In[1056]:


data[["returnb&h", "strategy"]].sum()


# In[1057]:


data[["returnb&h", "strategy"]].sum().apply(np.exp)


# In[1058]:


data[["returnb&h", "strategy"]].std()*np.sqrt(252)


# In[1059]:


data.Close.plot(figsize=(12,8))


# Stragedy adjusted with long bias

# In[1060]:


data["position2"]=np.where(data["sma_s"]>data["sma_l"],1,0)


# In[1061]:


data["strategy2"]= data["returnb&h"] * data.position2.shift(1)


# In[1062]:


data.dropna(inplace= True)


# In[1063]:


data


# In[1064]:


data[["returnb&h","strategy2"]].sum()


# In[1065]:


data[["returnb&h","strategy2"]].sum().apply(np.exp)


# In[1066]:


data[["returnb&h", "strategy2"]].std()*np.sqrt(252)


# creating a function

# In[1089]:


def test_strategy(stock,start,end,SMA):
    df=yf.download(stock,start=start,end=end)
    data=df.Close.to_frame()
    data["return"]=np.log(data.Close.div(data.Close.shift(1)))
    data["SMA_S"]=data.Close.rolling(int(SMA[0])).mean()
    data["SMA_L"]=data.Close.rolling(int(SMA[1])).mean()
    data.dropna(inplace= True)
    
    data["position"]=np.where(data["SMA_S"] > data["SMA_L"],1,-1)
    data["strategy"]=data["return"]*data.position.shift(1)
    data.dropna(inplace=True)
    ret= np.exp(data["strategy"].sum())
    std= data["strategy"].std()*np.sqrt(252)
    
    return ret,std
                          
                           


# In[1096]:


test_strategy("SPY","2000-01-01","2020-01-01",(50,200))


# Create a class 

# In[1159]:


class SMABacktester():
    def __init__(self,symbol,SMA_S, SMA_L,start,end):
        self.symbol=symbol
        self.SMA_S=SMA_S
        self.SMA_L=SMA_L
        self.start=start
        self.end=end
        self.results= None
        self.get_data()
        
    def get_data(self):
        df= yf.download(self.symbol,start=self.start,end=self.end)
        data=df.Close.to_frame()
        data["return"]=np.log(data.Close.div(data.Close.shift(1)))
        data["SMA_S"]=data.Close.rolling(self.SMA_S).mean()
        data["SMA_L"]=data.Close.rolling(self.SMA_L).mean()
        data.dropna(inplace= True)
        self.data2=data
        
        return data
    
    def test_results(self):
        data=self.data2.copy().dropna()
        data["position"]=np.where(data["SMA_S"] > data["SMA_L"],1,-1)
        data["strategy"]=data["return"]*data.position.shift(1)
        data.dropna(inplace=True)
        data["returnb&h"]= data["return"].cumsum().apply(np.exp)
        data["returnstrategy"]= data["strategy"].cumsum().apply(np.exp)
        perf=data["returnstrategy"].iloc[-1]
        outperf=perf-data["returnb&h"].iloc[-1]
        self.results=data
        
        ret= np.exp(data["strategy"].sum())
        std= data["strategy"].std()*np.sqrt(252)
        
        #return ret,std
        return round(perf,6), round(outperf,6)
    
    def plot_results(self):
        if self.results is None:
            print("Run the test please")
        else:
            title:"{}| SMA_S() | SMA_L()".format(self.symbol,self.SMA_S,self.SMA_L)
            self.results[["returnb&h","returnstrategy"]].plot( figsize=(12,8))
        
    
        
           


# In[1160]:


tester= SMABacktester("SPY",50,100,"2000-01-01","2023-01-01")


# In[1161]:


tester.test_results()


# In[1162]:


tester.plot_results()


# In[ ]:




