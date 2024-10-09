import pandas as pd
import plotly_express as px
bitcoin_df = pd.read_csv('BCHAIN-MKPRU.csv')
gold_df = pd.read_csv('LBMA-GOLD.csv')

# [cash, gold, bitcoin]
portfolio = [1000, 0, 0]
gold_commission = .01
bitcoin_commission = .02

#final portfolio if buy bitcoin on first day and sell on last day
simple_bitcoin_buy = portfolio[0]*(1-bitcoin_commission) / bitcoin_df.iloc[0, -1] # number of bitcoins bought
portfolio = [0, simple_bitcoin_buy, 0] # change cash to 0 and bitcoins to the number bought
simple_bitcoin_sell = portfolio[1]*bitcoin_df.iloc[-1, -1]*(1-bitcoin_commission) # cash made from selling bitcoins on final day
portfolio = [simple_bitcoin_sell, 0, 0] # change cash and bitcoins for final portfolio
print(portfolio)

#final portfolio if buy gold on first day and sell on last day
portfolio = [1000, 0, 0]
simple_gold_buy = portfolio[0]*(1-gold_commission) / gold_df.iloc[0, -1]
portfolio = [0, simple_gold_buy, 0]
simple_gold_sell = portfolio[1]*gold_df.iloc[-1, -1]*(1-gold_commission)
portfolio = [simple_gold_sell, 0, 0]
print(portfolio)

portfolio = [1000, 0, 0]
# create new column in each dataframe to store the number of standard deviations each value is from the mean of the 20 values prior
gold_df['Z_score'] = (gold_df['USD (PM)'] - gold_df['USD (PM)'].rolling(window=20).mean()) / gold_df['USD (PM)'].rolling(window=20).std()
bitcoin_df['Z_score'] = (bitcoin_df['Value'] - bitcoin_df['Value'].rolling(window=20).mean()) / bitcoin_df['Value'].rolling(window=20).std()

# the following code iterates through many combinations of three different parameters using the mean reversion strategy (only buy/selling gold) and tracks the final cash value of each portfolio
#this is meant to be a starting point. My other ideas to improve model include: using rolling window as a parameter, trying other values for parameters,
# testing reverse of mean reversion: buy if price is higher than normal and sell if price is lower than normal, buy/selling mix of gold and bitcoin each day,
#charting graph of parameter vs portfolio value to find optimal parameter value, and analyzing when algorithm buy/sells on a graph to see where it could improve.

final_values = [] # this will store final cash value of each portfolio from each combination of the three parameters
threshold_values = [1, 1.5, 2, 2.5, 3]  # different thresholds for standard deviations that indicate buy/sell signal
buy_weights = [.2, .5, .7, .9] # percent of cash in portfolio to use to buy gold once buy signal is triggered
sell_weights = [.2, .5, .7, .9] # percent of gold in portfolio to sell once sell signal is triggered

#iterate through each combination of thresholds and buy and sell weights
for threshold in threshold_values:
    for buy_weight in buy_weights:
        for sell_weight in sell_weights:
            
            portfolio = [1000, 0, 0]
            # iterate through each day and decide whether to buy, sell, or hold
            for i in range(1, len(gold_df)):
                if gold_df['Z_score'].iloc[i] < -threshold: # buy signal
                    # buy gold
                    amount_to_spend = buy_weight * portfolio[0] # amount of cash being used to buy gold
                    gold_bought = (amount_to_spend / gold_df['USD (PM)'].iloc[i]) * (1 - gold_commission) # oz of gold being purchased
                    portfolio[0] -= amount_to_spend # decrease cash in portfolio by amount spent on gold
                    portfolio[1] += gold_bought # increase gold in portfolio by amount purchased
                elif gold_df['Z_score'].iloc[i] > threshold: #sell signal
                    # sell gold
                    amount_to_sell = sell_weight * portfolio[1] # oz of gold getting sold
                    gold_sold = amount_to_sell * gold_df['USD (PM)'].iloc[i] * (1 - gold_commission) # cash generated from selling gold
                    portfolio[0] += gold_sold # increase cash in portfolio by amount made from selling gold
                    portfolio[1] -= amount_to_sell # decrease gold in portfolio by oz sold
            gold_sold = portfolio[1] * gold_df['USD (PM)'].iloc[i] * (1 - gold_commission) # sell all gold on final day
            portfolio[0] += gold_sold # increase cash in portfolio by amount made from selling gold
            portfolio[1] = 0 # decrease gold in portfolio to zero
            final_values.append(portfolio[0]) # add final cash value to the list
print(final_values)


# use the code below to plot value of gold and portfolio over time on same graph for a given set of parameters
'''total_money = [1000]
portfolio = [1000, 0, 0]
threshold = 2
buy_weight = .5
sell_weight = 1

for i in range(1, len(gold_df)):
    if gold_df['Z_score'].iloc[i] < -threshold:
        
        amount_to_spend = buy_weight * portfolio[0]
        gold_bought = (amount_to_spend / gold_df['USD (PM)'].iloc[i]) * (1 - gold_commission)
        portfolio[0] -= amount_to_spend
        portfolio[1] += gold_bought
    elif gold_df['Z_score'].iloc[i] > threshold:
        
        amount_to_sell = sell_weight * portfolio[1]
        gold_sold = amount_to_sell * gold_df['USD (PM)'].iloc[i] * (1 - gold_commission)
        portfolio[0] += gold_sold
        portfolio[1] -= amount_to_sell
    total_money.append(portfolio[0] + portfolio[1]*gold_df['USD (PM)'].iloc[i])
amount_to_sell = portfolio[1]
gold_sold = amount_to_sell * gold_df['USD (PM)'].iloc[i] * (1 - gold_commission)
portfolio[0] += gold_sold
portfolio[1] -= amount_to_sell
print(portfolio)
gold_df['portfolio'] = total_money
fig = px.line(gold_df, x='Date', y='USD (PM)')
fig.add_scatter(x=gold_df['Date'], y=gold_df['portfolio'])
fig.show()'''

