import pandas as pd
import numpy as np
import plotly_express as px
from datetime import datetime
import holidays
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


bitcoin_df = pd.read_csv('BCHAIN-MKPRU.csv')
gold_df = pd.read_csv('LBMA-GOLD.csv')


#print(bitcoin_df)
#print(gold_df)
# [cash, gold, bitcoin]
portfolio = [1000, 0, 0]
gold_commission = .01
bitcoin_commission = .02

#final portfolio if buy bitcoin on first day and sell on last day
def simple_bitcoin(bitcoin_commission: float) -> list:
    portfolio = [1000, 0, 0]
    simple_bitcoin_buy = portfolio[0]*(1-bitcoin_commission) / bitcoin_df.iloc[0, -1] # number of bitcoins bought
    portfolio = [0, simple_bitcoin_buy, 0] # change cash to 0 and bitcoins to the number bought
    simple_bitcoin_sell = portfolio[1]*bitcoin_df.iloc[-1, -1]*(1-bitcoin_commission) # cash made from selling bitcoins on final day
    portfolio = [simple_bitcoin_sell, 0, 0] # change cash and bitcoins for final portfolio
    return(portfolio)

#final portfolio if buy gold on first day and sell on last day
def simple_gold(gold_commisssion: float) -> list:
    portfolio = [1000, 0, 0]
    simple_gold_buy = portfolio[0]*(1-gold_commission) / gold_df.iloc[0, -1]
    portfolio = [0, simple_gold_buy, 0]
    simple_gold_sell = portfolio[1]*gold_df.iloc[-1, -1]*(1-gold_commission)
    portfolio = [simple_gold_sell, 0, 0]
    return(portfolio)


#best values: 6, 101. other good values: 2, 127
def test_moving_avg_values(short_low: int, short_high: int, long_low: int, long_high: int, threshold: int, asset: str) -> list:
    if asset == 'bitcoin':
        asset_df = bitcoin_df
        val = 'Value'
        commission = bitcoin_commission
    else:
        asset_df = gold_df
        val = 'USD (PM)'
        commission = gold_commission
    final_values = []
    short_list = []
    long_list = []
    portfolio_list = []
    for val1 in range(short_low, short_high):
        for val2 in range(long_low, long_high):
            portfolio = [1000, 0, 0]
            asset_df['short'] = asset_df[val].rolling(window=val1).mean()
            asset_df['long'] = asset_df[val].rolling(window=val2).mean()
            for i in range(0, len(asset_df)):
                if asset_df['short'].iloc[i] > asset_df['long'].iloc[i]:
                    if portfolio[0] > 0:
                        asset_bought = (portfolio[0] / asset_df[val].iloc[i]) * (1 - commission)
                        portfolio[0] = 0
                        portfolio[2] += asset_bought
                elif asset_df['short'].iloc[i] < asset_df['long'].iloc[i]:
                    if portfolio[2] > 0:
                        asset_sold = portfolio[2] * asset_df[val].iloc[i] * (1 - commission)
                        portfolio[0] += asset_sold
                        portfolio[2] = 0
            asset_sold = portfolio[2] * asset_df[val].iloc[-1] * (1 - commission)
            portfolio[0] += asset_sold
            portfolio[2] = 0
            portfolio[0] = round(portfolio[0])
            if portfolio[0] > threshold:
                final_values.append((portfolio[0], val1, val2))
            short_list.append(val1)
            long_list.append(val2)
            portfolio_list.append(portfolio[0])
    return(final_values)
    # 3D Plotting
    '''fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(short_list, long_list, portfolio_list, c=portfolio_list, cmap='viridis')
    ax.set_title('3D Plot of Rolling Average Values vs Final Portfolio Value', fontsize=14)
    ax.set_xlabel('Short Rolling Average Value', fontsize=12)
    ax.set_ylabel('Long Rolling Average Value', fontsize=12)
    ax.set_zlabel('Final Portfolio Value', fontsize=12)
    plt.show()'''

def compare_moving_avg_values_vs_commission(short1: int, long1: int, short2: int, long2: int, low: float, high: float, step: float):
    par1_end_values = []
    par2_end_values = []
    commissions = []
    for commission in range(int(high/step)):
        commission = commission * step
        commissions.append(commission)
        portfolio = [1000, 0, 0]
        bitcoin_df['short'] = bitcoin_df['Value'].rolling(window=short1).mean()
        bitcoin_df['long'] = bitcoin_df['Value'].rolling(window=long1).mean()
        for i in range(0, len(bitcoin_df)):
            if bitcoin_df['short'].iloc[i] > bitcoin_df['long'].iloc[i]:
                if portfolio[0] > 0:
                    bitcoin_bought = (portfolio[0] / bitcoin_df['Value'].iloc[i]) * (1 - commission)
                    portfolio[0] = 0
                    portfolio[2] += bitcoin_bought
            elif bitcoin_df['short'].iloc[i] < bitcoin_df['long'].iloc[i]:
                if portfolio[2] > 0:
                    bitcoin_sold = portfolio[2] * bitcoin_df['Value'].iloc[i] * (1 - commission)
                    portfolio[0] += bitcoin_sold
                    portfolio[2] = 0
        bitcoin_sold = portfolio[2] * bitcoin_df['Value'].iloc[-1] * (1 - commission)
        portfolio[0] += bitcoin_sold
        portfolio[2] = 0
        portfolio[0] = round(portfolio[0])
        par1_end_values.append(portfolio[0])

        portfolio = [1000, 0, 0]
        bitcoin_df['short'] = bitcoin_df['Value'].rolling(window=short2).mean()
        bitcoin_df['long'] = bitcoin_df['Value'].rolling(window=long2).mean()
        for i in range(0, len(bitcoin_df)):
            if bitcoin_df['short'].iloc[i] > bitcoin_df['long'].iloc[i]:
                if portfolio[0] > 0:
                    bitcoin_bought = (portfolio[0] / bitcoin_df['Value'].iloc[i]) * (1 - commission)
                    portfolio[0] = 0
                    portfolio[2] += bitcoin_bought
            elif bitcoin_df['short'].iloc[i] < bitcoin_df['long'].iloc[i]:
                if portfolio[2] > 0:
                    bitcoin_sold = portfolio[2] * bitcoin_df['Value'].iloc[i] * (1 - commission)
                    portfolio[0] += bitcoin_sold
                    portfolio[2] = 0
        bitcoin_sold = portfolio[2] * bitcoin_df['Value'].iloc[-1] * (1 - commission)
        portfolio[0] += bitcoin_sold
        portfolio[2] = 0
        portfolio[0] = round(portfolio[0])
        par2_end_values.append(portfolio[0])
    fig = px.line(x=commissions, y=par1_end_values)
    fig.add_scatter(x=commissions, y=par2_end_values)
    fig.show()

def plot_moving_avg_results(short: int, long: int, asset: str) -> list:
    if asset == 'bitcoin':
        asset_df = bitcoin_df
        val = 'Value'
        commission = bitcoin_commission
    else:
        asset_df = gold_df
        val = 'USD (PM)'
        commission = gold_commission
    portfolio = [1000, 0, 0]
    final_values = []
    asset_df['short'] = asset_df[val].rolling(window=short).mean()
    asset_df['long'] = asset_df[val].rolling(window=long).mean()
    for i in range(0, len(asset_df)):
        if asset_df['short'].iloc[i] > asset_df['long'].iloc[i]:
            if portfolio[0] > 0:
                asset_bought = (portfolio[0] / asset_df[val].iloc[i]) * (1 - commission)
                portfolio[0] = 0
                portfolio[2] += asset_bought
        elif asset_df['short'].iloc[i] < asset_df['long'].iloc[i]:
            if portfolio[2] > 0:
                asset_sold = portfolio[2] * asset_df[val].iloc[i] * (1 - commission)
                portfolio[0] += asset_sold
                portfolio[2] = 0
        final_values.append(portfolio[0] + portfolio[2]*asset_df[val].iloc[i])
        
    asset_sold = portfolio[2] * asset_df[val].iloc[-1] * (1 - commission)
    portfolio[0] += asset_sold
    portfolio[2] = 0
    portfolio[0] = round(portfolio[0])
    asset_df['portfolio'] = final_values
    fig = px.line(asset_df, x='Date', y=val)
    fig.add_scatter(x=asset_df['Date'], y=asset_df['short'])
    fig.add_scatter(x=asset_df['Date'], y=asset_df['long'])
    fig.add_scatter(x=asset_df['Date'], y=asset_df['portfolio'])
    fig.show()
    return(portfolio[0])

#print(test_bitcoin_moving_avg_values(1, 40, 70, 130, 1000, 'gold'))
#print(test_moving_avg_values(10, 40, 100, 180, 1380, 'gold'))
print(plot_moving_avg_results(23, 153, 'gold'))






# create new column in each dataframe to store the number of standard deviations each value is from the mean of the 20 values prior

#gold_df['Z_score'] = (gold_df['USD (PM)'] - gold_df['USD (PM)'].rolling(window=20).mean()) / gold_df['USD (PM)'].rolling(window=20).std()
#bitcoin_df['Z_score'] = (bitcoin_df['Value'] - bitcoin_df['Value'].rolling(window=20).mean()) / bitcoin_df['Value'].rolling(window=20).std()

# the following code iterates through many combinations of three different parameters using the mean reversion strategy (only buy/selling gold) and tracks the final cash value of each portfolio
#this is meant to be a starting point. My other ideas to improve model include: using rolling window as a parameter, trying other values for parameters,
# testing reverse of mean reversion: buy if price is higher than normal and sell if price is lower than normal, buy/selling mix of gold and bitcoin each day,
#charting graph of parameter vs portfolio value to find optimal parameter value, and analyzing when algorithm buy/sells on a graph to see where it could improve.



# use the code below to plot value of gold and portfolio over time on same graph for a given set of parameters
'''final_values = []
total_money = [1000]
portfolio = [1000, 0, 0]
b_buy_thresholds = [-1, 0, 1]
b_sell_thresholds = [4.5, 5, 5.5]
b_buy_weights = [.7, .9, 1]
b_sell_weights = [0, .2, .3]
g_buy_thresholds = [-2, -2.5, -3]
g_sell_thresholds = [2.5, 3, 3.5]
g_buy_weights = [0, .1, .3]
g_sell_weights = [.8, .9, 1]
weekend_days = 0
us_holidays = holidays.US()

for b_buy_threshold in b_buy_thresholds:
    for b_sell_threshold in b_sell_thresholds:
        for b_buy_weight in b_buy_weights:
            for b_sell_weight in b_sell_weights:
                for g_buy_threshold in g_buy_thresholds:
                    for g_sell_threshold in g_sell_thresholds:
                        for g_buy_weight in g_buy_weights:
                            for g_sell_weight in g_sell_weights:

                                portfolio = [1000, 0, 0]
                                weekend_days = 0
                                for i in range(1, len(bitcoin_df)):
                                    if datetime.strptime(bitcoin_df['Date'].iloc[i], '%m/%d/%y').weekday() < 5 and not us_holidays.is_working_day(datetime.strptime(bitcoin_df['Date'].iloc[i], '%m/%d/%y')):
                                        if bitcoin_df['Z_score'].iloc[i] < b_buy_threshold:
        
                                            amount_to_spend = b_buy_weight * portfolio[0]
                                            bitcoin_bought = (amount_to_spend / bitcoin_df['Value'].iloc[i]) * (1 - bitcoin_commission)
                                            portfolio[0] -= amount_to_spend
                                            portfolio[2] += bitcoin_bought
                                        elif bitcoin_df['Z_score'].iloc[i] > b_sell_threshold:
        
                                            amount_to_sell = b_sell_weight * portfolio[2]
                                            bitcoin_sold = amount_to_sell * bitcoin_df['Value'].iloc[i] * (1 - bitcoin_commission)
                                            portfolio[0] += bitcoin_sold
                                            portfolio[2] -= amount_to_sell
    
                                        if gold_df['Z_score'].iloc[i-weekend_days] < g_buy_threshold:
        
                                            amount_to_spend = g_buy_weight * portfolio[0]
                                            gold_bought = (amount_to_spend / gold_df['USD (PM)'].iloc[i-weekend_days]) * (1 - gold_commission)
                                            portfolio[0] -= amount_to_spend
                                            portfolio[1] += gold_bought
                                        elif gold_df['Z_score'].iloc[i-weekend_days] > g_sell_threshold:
        
                                            amount_to_sell = g_sell_weight * portfolio[1]
                                            gold_sold = amount_to_sell * gold_df['USD (PM)'].iloc[i-weekend_days] * (1 - bitcoin_commission)
                                            portfolio[0] += gold_sold
                                            portfolio[1] -= amount_to_sell
                                    else:
                                        weekend_days += 1
                                        if bitcoin_df['Z_score'].iloc[i] < b_buy_threshold:
        
                                            amount_to_spend = b_buy_weight * portfolio[0]
                                            bitcoin_bought = (amount_to_spend / bitcoin_df['Value'].iloc[i]) * (1 - bitcoin_commission)
                                            portfolio[0] -= amount_to_spend
                                            portfolio[2] += bitcoin_bought
                                        elif bitcoin_df['Z_score'].iloc[i] > b_sell_threshold:
        
                                            amount_to_sell = b_sell_weight * portfolio[2]
                                            bitcoin_sold = amount_to_sell * bitcoin_df['Value'].iloc[i] * (1 - bitcoin_commission)
                                            portfolio[0] += bitcoin_sold
                                            portfolio[2] -= amount_to_sell
                                bitcoin_to_sell = portfolio[2]
                                bitcoin_sold = bitcoin_to_sell * bitcoin_df['Value'].iloc[i] * (1 - bitcoin_commission)
                                portfolio[0] += bitcoin_sold
                                portfolio[2] -= bitcoin_to_sell
                                gold_to_sell = portfolio[1]
                                gold_sold = gold_to_sell * gold_df['USD (PM)'].iloc[i-weekend_days] * (1 - bitcoin_commission)
                                portfolio[0] += gold_sold
                                portfolio[0] = round(portfolio[0])
                                portfolio[1] -= gold_to_sell
                                if portfolio[0] > 73000:
                                    final_values.append((portfolio[0], b_buy_threshold, b_sell_threshold, b_buy_weight, b_sell_weight, g_buy_threshold, g_sell_threshold, g_buy_weight, g_sell_weight))
                                #total_money.append(portfolio[0] + portfolio[1]*gold_df['USD (PM)'].iloc[i-weekend_days] + portfolio[2]*bitcoin_df['Value'].iloc[i])
print(final_values)'''

'''position = [0]
total_money = [1000]
portfolio = [1000, 0, 0]
b_buy_threshold = 1
b_sell_threshold = 5
b_buy_weight = 1
b_sell_weight = .2
g_buy_threshold = -2.5
g_sell_threshold = 3
g_buy_weight = 0
g_sell_weight = .8
weekend_days = 0
transactions = 0
us_holidays = holidays.US()
for i in range(1, len(bitcoin_df)):
    if datetime.strptime(bitcoin_df['Date'].iloc[i], '%m/%d/%y').weekday() < 5 and not us_holidays.is_working_day(datetime.strptime(bitcoin_df['Date'].iloc[i], '%m/%d/%y')):
        if bitcoin_df['Z_score'].iloc[i] < b_buy_threshold:
        
            amount_to_spend = b_buy_weight * portfolio[0]
            bitcoin_bought = (amount_to_spend / bitcoin_df['Value'].iloc[i]) * (1 - bitcoin_commission)
            portfolio[0] -= amount_to_spend
            portfolio[2] += bitcoin_bought
            transactions += 1
        elif bitcoin_df['Z_score'].iloc[i] > b_sell_threshold:
        
            amount_to_sell = b_sell_weight * portfolio[2]
            bitcoin_sold = amount_to_sell * bitcoin_df['Value'].iloc[i] * (1 - bitcoin_commission)
            portfolio[0] += bitcoin_sold
            portfolio[2] -= amount_to_sell
            transactions += 1
        if gold_df['Z_score'].iloc[i-weekend_days] < g_buy_threshold:
        
            amount_to_spend = g_buy_weight * portfolio[0]
            gold_bought = (amount_to_spend / gold_df['USD (PM)'].iloc[i-weekend_days]) * (1 - gold_commission)
            portfolio[0] -= amount_to_spend
            portfolio[1] += gold_bought
        elif gold_df['Z_score'].iloc[i-weekend_days] > g_sell_threshold:
        
            amount_to_sell = g_sell_weight * portfolio[1]
            gold_sold = amount_to_sell * gold_df['USD (PM)'].iloc[i-weekend_days] * (1 - bitcoin_commission)
            portfolio[0] += gold_sold
            portfolio[1] -= amount_to_sell
    else:
        weekend_days += 1
        if bitcoin_df['Z_score'].iloc[i] < b_buy_threshold:
        
            amount_to_spend = b_buy_weight * portfolio[0]
            bitcoin_bought = (amount_to_spend / bitcoin_df['Value'].iloc[i]) * (1 - bitcoin_commission)
            portfolio[0] -= amount_to_spend
            portfolio[2] += bitcoin_bought
            transactions += 1
        elif bitcoin_df['Z_score'].iloc[i] > b_sell_threshold:
        
            amount_to_sell = b_sell_weight * portfolio[2]
            bitcoin_sold = amount_to_sell * bitcoin_df['Value'].iloc[i] * (1 - bitcoin_commission)
            portfolio[0] += bitcoin_sold
            portfolio[2] -= amount_to_sell
            transactions += 1
    total_money.append(portfolio[0] + portfolio[1]*gold_df['USD (PM)'].iloc[i-weekend_days] + portfolio[2]*bitcoin_df['Value'].iloc[i])
                                
bitcoin_to_sell = portfolio[2]
bitcoin_sold = bitcoin_to_sell * bitcoin_df['Value'].iloc[i] * (1 - bitcoin_commission)
portfolio[0] += bitcoin_sold
portfolio[2] -= bitcoin_to_sell
gold_to_sell = portfolio[1]
gold_sold = gold_to_sell * gold_df['USD (PM)'].iloc[i-weekend_days] * (1 - bitcoin_commission)
portfolio[0] += gold_sold
portfolio[1] -= gold_to_sell
print(portfolio)
print(transactions)
bitcoin_df['portfolio'] = total_money
bitcoin_df['Value'] = bitcoin_df['Value'] * 1000 / bitcoin_df['Value'].iloc[0]
fig = px.line(bitcoin_df, x='Date', y='Value')
fig.add_scatter(x=bitcoin_df['Date'], y=bitcoin_df['portfolio'])
fig.show()'''

