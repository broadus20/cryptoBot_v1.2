import cbpro
import time 
import logging
import seaborn as sns
import pandas as pd
import telegram_send
import matplotlib.pyplot as plt

client = cbpro.PublicClient()


bull_market_detection = False

def MACD(DF, a=12,b=26,c=9):
    df = DF.copy()
    df["ma_fast"] = df['Close'].ewm(span=a, min_periods=a).mean()
    df["ma_slow"] = df['Close'].ewm(span=b, min_periods=b).mean()
    df['macd'] = df['ma_fast'] - df['ma_slow']
    df['signal'] = df['macd'].ewm(span=c, min_periods=c).mean()
    return df.loc[:,['macd', 'signal', 'ma_fast', 'ma_slow']]

def SMA(df,a,b):
    "function to calculate SMA"
    df['sma_fast']=df['Close'].rolling(a).mean()
    df['sma_slow']=df['Close'].rolling(b).mean()
    return df

def ATR(DF,n):
    "function to calculate True Range and Average True Range"
    df = DF.copy()
    df['H-L']=abs(df['High']-df['Low'])
    df['H-PC']=abs(df['High']-df['Close'].shift(1))
    df['L-PC']=abs(df['Low']-df['Close'].shift(1))
    df['TR']=df[['H-L','H-PC','L-PC']].max(axis=1,skipna=False)
    df['ATR'] = df['TR'].rolling(n).mean()
    #df['ATR'] = df['TR'].ewm(span=n,adjust=False,min_periods=n).mean()
    df2 = df.drop(['H-L','H-PC','L-PC'],axis=1)
    return round(df2["ATR"][-1],2)    

# Dataframe generation
def get_crypto_data(market):
    '''Returns DFs for [x, "15min" "1h", "6h", "1d"] with columns'''

    client = cbpro.PublicClient()
    # Min Data
    BTC_data = pd.DataFrame(client.get_product_historic_rates(market, start=None, end=None,
                                                              granularity=None))  # BTC Data granularity desired time slice in seconds
    BTC_data.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    BTC_data['Date'] = pd.to_datetime(BTC_data['Date'], unit='s')
    BTC_data.set_index('Date', inplace=True)
    BTC_data.sort_values(by='Date', ascending=True, inplace=True)
    BTC_data['Return'] = BTC_data['Close'].pct_change()

    # 15min Data
    BTC_15min_data = pd.DataFrame(client.get_product_historic_rates(market, start=None, end=None,
                                                                    granularity=900))
    BTC_15min_data.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    BTC_15min_data['Date'] = pd.to_datetime(BTC_15min_data['Date'], unit='s')
    BTC_15min_data.set_index('Date', inplace=True)
    BTC_15min_data.sort_values(by='Date', ascending=True, inplace=True)
    BTC_15min_data[['MACD', "SIGNAL", 'ma_fast', 'ma_slow']] = MACD(BTC_15min_data)
    BTC_15min_data.fillna(0, inplace=True)
    BTC_15min_data['Return'] = BTC_15min_data['Close'].pct_change()

    # Hour Data
    BTC_hour_data = pd.DataFrame(client.get_product_historic_rates(market, start=None, end=None,
                                                                   granularity=3600))
    BTC_hour_data.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    BTC_hour_data['Date'] = pd.to_datetime(BTC_hour_data['Date'], unit='s')
    BTC_hour_data.set_index('Date', inplace=True)
    BTC_hour_data.sort_values(by='Date', ascending=True, inplace=True)
    BTC_hour_data[['MACD', "SIGNAL", 'ma_fast', 'ma_slow']] = MACD(BTC_hour_data)
    BTC_hour_data.fillna(0, inplace=True)
    BTC_hour_data['Return'] = BTC_hour_data['Close'].pct_change()

    # 6 Hour Data
    BTC_6hour_data = pd.DataFrame(client.get_product_historic_rates(market, start=None, end=None,
                                                                    granularity=21600))
    BTC_6hour_data.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    BTC_6hour_data['Date'] = pd.to_datetime(BTC_6hour_data['Date'], unit='s')
    BTC_6hour_data.set_index('Date', inplace=True)
    BTC_6hour_data.sort_values(by='Date', ascending=True, inplace=True)
    BTC_6hour_data[['MACD', "SIGNAL", 'ma_fast', 'ma_slow']] = MACD(BTC_6hour_data)
    BTC_6hour_data.fillna(0, inplace=True)
    BTC_6hour_data['Return'] = BTC_6hour_data['Close'].pct_change()

    # Daily Data
    BTC_daily_data = pd.DataFrame(client.get_product_historic_rates(market, start=None, end=None,
                                                                    granularity=86400))
    BTC_daily_data.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    BTC_daily_data['Date'] = pd.to_datetime(BTC_daily_data['Date'], unit='s')
    BTC_daily_data.set_index('Date', inplace=True)
    BTC_daily_data.sort_values(by='Date', ascending=True, inplace=True)
    BTC_daily_data[['MACD', "SIGNAL", 'ma_fast', 'ma_slow']] = MACD(BTC_daily_data)
    BTC_daily_data.fillna(0, inplace=True)
    BTC_daily_data['Return'] = BTC_daily_data['Close'].pct_change()
    
    return BTC_data, BTC_15min_data, BTC_hour_data, BTC_6hour_data, BTC_daily_data


def get_condition():
    if BTC_hour_data['ma_fast'].iloc[-1] > BTC_hour_data['ma_slow'].iloc[-1] and BTC_6hour_data['ma_fast'].iloc[-1] > \
            BTC_6hour_data['ma_slow'].iloc[-1]:
        logger.info('returning 15min df')
        return BTC_15min_data
    elif BTC_hour_data['ma_fast'].iloc[-1] < BTC_hour_data['ma_slow'].iloc[-1] and BTC_6hour_data['ma_fast'].iloc[-1] < \
            BTC_6hour_data['ma_slow'].iloc[-1]:
        logger.info('returning hour df')
        return BTC_hour_data
    else:
        logger.info('returning final hour df')
        return BTC_hour_data


def trade_signal(df, market):
    "function to generate signal"
    global upward_sma_dir, dnward_sma_dir
    signal = ""
    if df['ma_fast'][-1] > df['ma_slow'][-1] and df['ma_fast'][-2] < df['ma_slow'][-2]:
        upward_ema_dir[market] = True
        dnward_ema_dir[market] = False
    if df['ma_fast'][-1] < df['ma_slow'][-1] and df['ma_fast'][-2] > df['ma_slow'][-2]:
        upward_ema_dir[market] = False
        dnward_ema_dir[market] = True
    if df['MACD'][-1] > df['SIGNAL'][-1] and df['MACD'][-2] < df['SIGNAL'][-2]:
            upward_macd_dir[market] = True
            dnward_macd_dir[market] = False
    if df['MACD'][-1] < df['SIGNAL'][-1] and df['MACD'][-2] > df['SIGNAL'][-2]:
            upward_macd_dir[market] = False
            dnward_macd_dir[market] = True
    if (upward_ema_dir[market] ==True and df['MACD'][-1] > df['SIGNAL'][-1]) or (upward_macd_dir[market] == True and df['ma_fast'][-1] > df['ma_slow'][-1]): #Buy Signal
        if bull_market_detection == False:
            signal = "Buy"
        else:
            if df['sma_fast'][-1] > df['sma_slow'][-1]:#golden cross to determine bull market
                signal = "Buy"
    if (dnward_ema_dir[market] == True and df['MACD'][-1] < df['SIGNAL'][-1]) or (dnward_macd_dir[market] == True and df['ma_fast'][-1] < df['ma_slow'][-1]): #Sell Signal
        signal = "Sell"

    fig, ax1 = plt.subplots(ncols=1, figsize=(12, 4))
    sns.lineplot(data=df, x='Date', y='SIGNAL', ax=ax1).set(title=f"{market} MACD v. Signal")
    sns.lineplot(data=df, x='Date', y='MACD', ax=ax1)
    plt.legend(labels=['SIGNAL', 'MACD'], loc='upper left')
    plt.savefig('MACD_SIGNAL.png')

    fig, ax1 = plt.subplots(ncols=1, figsize=(12, 4))
    sns.lineplot(data=df.iloc[-200:], x='Date', y='Close', ax=ax1, label = 'close').set(title=f"{market} EMA Crossover")
    sns.lineplot(data=df.iloc[-200:], x='Date', y='ma_fast', ax=ax1, label = 'ema_12')
    sns.lineplot(data=df.iloc[-200:], x='Date', y='ma_slow', ax=ax1, label = 'ema_26')
    plt.legend(labels=['close', 'ema_12', 'ema_26'], loc='upper left')
    plt.savefig('ema.png')

    return signal

class CryptoBot:
    '''
    Main level for crypto bot
    ############################################################################################
    __init__(self):
        self.market = 'BTC-USD'/'ETH-USE/'SOL-USD'/'DOGE-USD'/
        self.signal= 'Buy'/'Sell'/''
        self.BTC_data, self.BTC_15min_data, self.BTC_hour_data, self.BTC_6hour_data, self.BTC_daily_data
        self.df1
        self.pos_size= int
        self.market_price= float
        self.sl= float
    market_order(self, market, side, sl, funds=pos_size, overdraft_enabled=False):
    trade_signal(self, df, market):
    plot_coin_price(self, DF):
    plot_MACD_signal(self, DF):
    plot_return(self, DF):
    '''
    def __init__(self):
        #self.markets = ['BTC-USD', 'SOL-USD'] #Iterate through others in main cbpro
        #self.market = self.markets[0]
        #self.signal=''
        #self.buy_sell_marker = ''
        #self.BTC_data, self.BTC_15min_data, self.BTC_hour_data, self.BTC_6hour_data, self.BTC_daily_data = get_crypto_data(market=self.market)
        #self.df1 = DataFrame.()

        self.pos_size = 10
        self.market_price = 0
        self.sl = 0
        
    def market_order(self, market, side, sl=0, funds=0, overdraft_enabled=False):
        """units can be positive or negative, stop loss (in pips) added/subtracted to price """
        if side == 'Buy':
            logger.debug(f"# Place market order and stoploss for {market}")
            try:
                crypto_size = round((pos_size/market_price),3)
                fill = auth_client.place_order(product_id=market, side='buy', order_type='market', size=crypto_size, overdraft_enabled=False)
                fill
                print(auth_client.get_order(fill['id']))
                crypto_size_sl = round((pos_size/market_price)-(.005*pos_size/market_price),3)
                fees= (.005*pos_size)
                pflio.update_order_book(fill) #UPDATE ORDER BOOK
                stop = auth_client.place_order(product_id=market, side='sell', order_type='limit', stop='loss', price=sl, stop_price=sl, size=crypto_size_sl)
                stop
                logger.info(f'{market} Market order successfully placed of {crypto_size}{market}  at _ with stoploss: {sl}')
            except:
                raise ValueError
        elif side == 'Sell':
            logger.debug(f"# Selling {market}")
            try:
                auth_client.cancel_all(product_id=market)
                coin_qty = pflio.get_coin_size(market) #Retrieve amount of Crypto held
                fill = auth_client.place_order(product_id=market, side='sell', order_type='market', size=coin_qty, overdraft_enabled=False)
                fill
                print(auth_client.get_order(fill['id']))
                pflio.update_order_book(fill) # Update Portfolio
                logger.info(f'{market} Market order sell successfully placed for {coin_qty}{market}')
                return
            except:
                raise ValueError

    def plot_coin_price(self, DF):
        #Visualize candle movement
        crypto_data = copy.deepcopy(DF)
        fig, ax1 = plt.subplots(ncols=1, figsize=(12, 4))
        _ = sns.lineplot(data=crypto_data, x='Date', y='Close', ax=ax1).set(title=f"Bitcoin Price Evolution")
        logger.info('printing plot coin price chart')
        return _

    def plot_MACD_signal(self, DF):
        # Visualize MCAD
        crypto_data = copy.deepcopy(DF)
        MACD_data = crypto_data[['MACD', 'SIGNAL']]
        fig, ax1 = plt.subplots(ncols=1, figsize=(12, 4))
        _ = sns.lineplot(data=MACD_data, ax=ax1).set(title=f"{BTC} EMA12 v. EMA26")
        logger.info('printing plot MACD signal chart')
        return _

    def plot_return(self, DF):
        #Visualize return / candle
        crypto_data = copy.deepcopy(DF)
        fig, ax1 = plt.subplots(ncols=1, figsize=(12,4))
        _ = sns.lineplot(data=crypto_data, x='Date', y='Return', ax=ax1).set(title=f"BTC Daily Return")
        logger.info('printing plot returns chart')
        return _


class Portfolio(object):
    '''
    A class to manage the data for your portfolio and order book:
    Order Book tracks all fills for sells and buys;
    Portfolio tracks current asset holdings and resembles get_account()
    Methods defined here:
        __init__(self)
        ---------------------------------------
    '''

    def __init__(self):
        logger.debug('establish key and dict for unfilled orders')
        self.DF = pd.DataFrame()
        self.m = 1  # 1 crypto in portfolio
        self.x = 0  # number of assets to remove
        self.unfilled_orders_dict = {}
        self.unfilled_orders_keys = []
        self.orderbook_keys = ["Date",
                               "Ex. Val",
                               "Fee",
                               "Filled Size",
                               "Funds",
                               "ID",
                               "Post",
                               "Market",
                               "Settled",
                               "Side",
                               "Size",
                               "Status",
                               "Stp",
                               "Type",]
        self.pflio_keys = ["ID",
                           "Currency",
                           "Size",
                           "Hold",
                           "Available",
                           "ProfileID",
                           "Trading Enabled"]
        self.USD = 0
        self.BTC = 0
        self.BTCUSD = self.BTC * float(list(client.get_product_ticker(product_id='BTC-USD').values())[1])
        self.SOL = 0
        self.SOLUSD = self.SOL * float(list(client.get_product_ticker(product_id='SOL-USD').values())[1])
        self.ETH = 0
        self.ETHUSD = self.ETH * float(list(client.get_product_ticker(product_id='ETH-USD').values())[1])
        self.DOGE = 0
        self.DOGEUSD = self.SOL * float(list(client.get_product_ticker(product_id='DOGE-USD').values())[1])

        self.portfolio = self.retrieve_pflio()
        self.orderbook = self.retrieve_orderbook()

    # function to calculate portfolio return iteratively
    def pflio(self):
        """Returns cumulative portfolio return
        DF = dataframe with monthly return info for all stocks
        m = number of stock in the portfolio
        x = number of underperforming stocks to be removed from portfolio monthly"""
        df = self.DF.copy()
        return

    def update_order_book(self, fill):
        self.unfilled_orders_dict[fill['id']] = fill
        temp_df = pd.DataFrame(self.unfilled_orders_dict).T
        #Remove fills with > 14 columns
        temp_df.columns = self.orderbook_keys
        for i in temp_df.columns:
            if i not in self.orderbook_keys:
                tmp.drop(i, axis=1, inplace=True)
        merged_df = temp_df.merge(self.orderbook, how="outer", on="Date")
        orderbook = merged_df
        return orderbook

    def update_pflio(self):
        logger.debug('Retrieve Current Pflio ')
        accountX = self.get_accounts()
        temp = {}
        for account in accountX:
            temp[account['currency']] = account
        temp_df = pd.DataFrame(temp).T
        self.portfolio = temp_df
        return self.portfolio

        # might want to add iterating through current portfolio and seeing what needs to be added and dropped
        # temp_pflio = self.retrieve_pflio()

    def retrieve_pflio(self):
        try:
            logger.debug('Retrieving portfolio from csv file in DIR')
            pflio = pd.read_csv("crypto_portfolio.csv", encoding="ISO-8859-1")
            return pflio

        except:
            logger.debug('Creating CSV file if none')
            portfolio = pd.DataFrame(columns=self.pflio_keys)
            portfolio.to_csv("crypto_portfolio.csv", index=False)
            pflio = pd.read_csv("crypto_portfolio.csv", encoding="ISO-8859-1")
            return pflio

    def retrieve_orderbook(self):
        try:
            logger.debug('Retrieving orderbook from csv file in DIR')
            order_book = pd.read_csv("crypto_orders.csv", encoding="ISO-8859-1")
            return orderbook
        except:
            logger.debug('Creating CSV file if none')
            orderbook = pd.DataFrame(columns=self.orderbook_keys)
            orderbook.to_csv("crypto_orders.csv", index=True)
            orderbook = pd.read_csv("crypto_orders.csv", encoding="ISO-8859-1")
            return orderbook

    def get_accounts(self):
        '''Returns accounts with crypto in them. Gather all data for currency in account'''
        auth_client = cbpro.AuthenticatedClient(Sandbox_KEY, Sandbox_SECRET, Sandbox_PASSPHRASE,
                                          api_url="https://api-public.sandbox.pro.coinbase.com")
        accounts = []
        account_ids = []
        coin_accounts = []
        for account in auth_client.get_accounts():
            keys = account.keys()
            for key in keys:
                if float(account['balance']) != 0.0:
                    accounts.append(account)
        for account in accounts:
            if account['id'] not in account_ids:
                account_ids.append(account['id'])
                coin_accounts.append(account)
        return coin_accounts

    def get_coin_size(self, market):
        '''Returns float in amount of crypto in account. Iterate through accounts the find size of coin'''
        accountX = self.get_accounts()
        for item in accountX:
            if item['currency'] == market[0:3]:
                return round(float(item['balance']), 8)

telegram_token = ''
chat_id = ''
path_config = telegram_send.get_config_path()
with open(path_config, 'w') as f:
    f.write(f'[telegram]\ntoken = {telegram_token}\nchat_id = {chat_id}')

class TelegramBot:
    '''
    Main lvl for telegram
    __init__(self)

    send_message(self):
        Return message of self.message to tele
    send_image(self, image):
        Returns input image to tele
    '''

    def __init__(self):
        self.token = ""
        self.config_file = ""

        self.cl_args = ""
        self.market = ""

        self.exchange = ""
        self.pair = ""
        self.overrides = ""

        self.message = 'Hello, I am Crypto Bot 3000'
        self.image = ''

    def send_message(self, message=''):
        '''input a message to send to tellebot'''
        logger.debug('message sending')
        telegram_send.send(messages=[self.message])
        # telegram-send --image moon.jpg --caption "The Moon at Night"

    def send_image(self, image):
        '''input a image to send to telebot'''
        logger.debug('image sending')
        with open(image, 'rb') as f:
            telegram_send.send(images=[f])


#Set Logging
logging.basicConfig(format='[%(levelname) 5s/%(asctime)s] %(name)s: %(message)s', level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

#Crypto Bot: Use (help(crypto_bot)) for help
cryptoBot = CryptoBot()
#Telegram Bot: use (help(tele_bot)) for help// teleBot.message to send message
teleBot = TelegramBot()

#Crypto Portfolio, call pflio.portfolio to see all entries in pflio
pflio = Portfolio()
pflio.update_pflio()

signal= ''
buy_sell_marker = ''
markets =  ['BTC-USD'] #'SOL-USD','ETH-USD','DOGE-USD'
pos_size = 100

upward_sma_dir = {}
dnward_sma_dir = {}
upward_ema_dir = {}
dnward_ema_dir = {}
upward_macd_dir = {}
dnward_macd_dir = {}
for i in markets:
    upward_sma_dir[i] = False
    dnward_sma_dir[i] = False
    upward_ema_dir[i] = False
    dnward_ema_dir[i] = False
    upward_macd_dir[i] = False
    dnward_macd_dir[i] = False
    
auth_client = cbpro.AuthenticatedClient(Sandbox_KEY, Sandbox_SECRET, Sandbox_PASSPHRASE,
                                  api_url="https://api-public.sandbox.pro.coinbase.com")
client = cbpro.PublicClient()

market = 'BTC-USD'

def main():
    global markets, buy_sell_marker, BTC_data, BTC_15min_data, BTC_hour_data, BTC_6hour_data, BTC_daily_data, df1 
    try:
        for market in markets:
            logger.info(f"Analysing {market}")
            BTC_data, BTC_15min_data, BTC_hour_data, BTC_6hour_data, BTC_daily_data = get_crypto_data(market=market)
            df1 = get_condition()
            df1 = SMA(df1, 50, 200) #Bull market detection
            cryptoBot.df1 = df1
            signal = trade_signal(cryptoBot.df1, market)
           #logger.info(f'Signal is: {signal}')
            if signal == 'Buy':
                if buy_sell_marker != 'Buy':
                    logger.info(f"Purchasing {market}...")
                    #place_order(market, signal, funds=pos_size, sl=3*ATR(df1,120))
                    market_price =  float(list(client.get_product_ticker(product_id= market).values())[1])
                    sl =round( market_price - 3*ATR(df1,120),2)
                    try:
                        cryptoBot.market_order(market, signal, sl, funds=pos_size)
                    except:
                        logging.critical("error")
                        pass
                    teleBot.message = f"Crypto Bot placing {market} market {signal} order  at:\
                        \nPrice: {market_price} \nStoploss: {sl}"
                    teleBot.send_message()
                    teleBot.send_image('MACD_SIGNAL.png')
                    teleBot.send_image('ema.png')
                    logger.info(f"Market order {signal} of ${pos_size}USD placed for {market} at: \nPrice: {market_price} \
                                \nStoploss: {sl}")
                    #set buy_sell_marker
                    buy_sell_marker = 'Buy'
                else:
                    logger.info(f'buy_sell_marker is set to {buy_sell_marker}')
            elif signal == 'Sell':
                if buy_sell_marker != 'Sell':
                    logger.info(f"Selling {market}")
                    market_price =  float(list(client.get_product_ticker(product_id= market).values())[1])
                    try:
                        cryptoBot.market_order(market, signal)
                    except:
                        pass
                    teleBot.message = f"Crypto Bot placing {market} market {signal} order at:\
                        \nPrice: {market_price}"
                    teleBot.send_message()
                    teleBot.send_image('MACD_SIGNAL.png')
                    teleBot.send_image('ema.png')
                    logger.info(f"Crypto Bot placing {market} market {signal} order of ${pos_size}USD at:\
                        \nPrice: {market_price}")
                    #set buy_sell_marker
                    buy_sell_marker = 'Sell'
                else:
                    logger.info(f'buy_sell_marker is set to {buy_sell_marker}')
        # Take profit and market reversal detection..........
    except:
        print("error encountered....skipping this iteration")
#Initialize
teleBot.message = f'CryptoBot initializing... \n analyzing... {markets}'
teleBot.send_message()
# Continuous execution
starttime = time.time()
timeout = time.time() + 60 * 60 * 1  # 60 seconds times 60 meaning the script will run for 1 hr * 6, 6 hours
main()
while time.time() <= timeout:
    try:
        time_d = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        logging.info("passthrough at " + time_d)
        print("passthrough at ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        main()
        time.sleep(900 - ((time.time() - starttime) % 900.0))  # 15 minute interval between each new execution, 900s
    except KeyboardInterrupt:
        print('\n\nKeyboard exception received. Exiting.')
