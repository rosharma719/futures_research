
#Packages
import pandas as pd
import numpy as np 
from talib import abstract
from talib.abstract import *
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.ensemble import GradientBoostingClassifier
import warnings

warnings.filterwarnings(action='ignore', category=FutureWarning, module='sklearn')

def backtest(ticker):

    df = pd.read_csv(f"D:\\RA_stock_prediction\\RA_stock_prediction\\data\\futures_full_1min_continuous_adjusted_kzvua9e\\{ticker}_1min_continuous_adjusted.txt", sep=',', header=None)
    
    # If the CSV does not have headers, uncomment the following line to manually set them
    #Setting inputs 
    inputs = {
        "open": df[1], 
        "high": df[2], 
        'low': df[3], 
        'close': df[4],
        'volume': df[5]
    }

    signals = pd.DataFrame(index=df.index)
    print(ticker)

    # Calculating indicators
    signals['EMA10'] = abstract.EMA(inputs, timeperiod=10)
    signals['SMA30'] = abstract.SMA(inputs, timeperiod=30)
    signals['MACD'], signals['MACD_signal'], _ = abstract.MACD(inputs, fastperiod=12, slowperiod=26, signalperiod=9)
    signals['upper_band'], signals['middle_band'], signals['lower_band'] = abstract.BBANDS(inputs, timeperiod=20, nbdevup=1.0, nbdevdn=1.0, matype=0)
    signals['RSI'] = abstract.RSI(inputs, timeperiod=14)
    signals['ROC'] = abstract.ROC(inputs, timeperiod=10)
    signals['VOL'] = df[4]
    signals['ADX'] = abstract.ADX(inputs, timeperiod=14)
    signals['ATR'] = abstract.ATR(inputs, timeperiod=14)
    signals['CMO'] = abstract.CMO(inputs, timeperiod=14)
    signals['OBV'] = abstract.OBV(inputs['close'], inputs['volume'])
    signals['WILLR'] = abstract.WILLR(inputs, timeperiod=14)
    signals['CCI'] = abstract.CCI(inputs, timeperiod=14)
    signals['MFI'] = abstract.MFI(inputs, timeperiod=14)
    signals['PPO'] = abstract.PPO(inputs, fastperiod=12, slowperiod=26, matype=0)
    signals['ULTOSC'] = abstract.ULTOSC(inputs, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    signals['CMF'] = abstract.ADOSC(inputs, fastperiod=3, slowperiod=10)
    signals['SAR'] = abstract.SAR(inputs, acceleration=0.02, maximum=0.2)
    signals['slowk'], signals['slowd'] = abstract.STOCH(inputs, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    signals['APO'] = abstract.APO(inputs, fastperiod=12, slowperiod=26)
    signals['AROON_down'], signals['AROON_up'] = abstract.AROON(inputs, timeperiod=14)
    signals['DEMA'] = abstract.DEMA(inputs, timeperiod=30)
    signals['HT_TRENDLINE'] = abstract.HT_TRENDLINE(inputs)
    signals['KAMA'] = abstract.KAMA(inputs, timeperiod=30)
    signals['MOM'] = abstract.MOM(inputs, timeperiod=10)
    signals['TEMA'] = abstract.TEMA(inputs, timeperiod=30)
    signals['TRIX'] = abstract.TRIX(inputs, timeperiod=15)


    EMA20 = abstract.EMA(inputs, timeperiod=20)
    ATR10 = abstract.ATR(inputs, timeperiod=10)
    signals['KC_upper'] = EMA20 + (ATR10 * 2)
    signals['KC_middle'] = EMA20
    signals['KC_lower'] = EMA20 - (ATR10 * 2)

    # Calculate the 2-hour (120 minutes) rolling maximum of the 'close' price
    signals['2hr_max_close'] = df[3].rolling(window=120, min_periods=1).max().shift(-120)

    # Define the threshold for the price increase 
    price_increase_threshold = 1.01  
    signals['target'] = (signals['2hr_max_close'] >= df[3] * price_increase_threshold).astype(int)

    num_buy_signals = signals['target'].sum()


    # Drop rows with NaN values
    clean_signals = signals.dropna()


    # Split data into features and target
    X = clean_signals[['EMA10', 'SMA30', 'MACD', 'MACD_signal', 'upper_band', 'middle_band', 'lower_band', 'RSI', 'ROC', 'VOL', 'ADX', 'ATR', 'CMO', 'OBV', 'WILLR', 'CCI', "MFI", "PPO", "ULTOSC", "CMF", "SAR", "slowk", "APO", "AROON_down", "DEMA", "HT_TRENDLINE", "KAMA", "MOM", "TEMA", "TRIX", "KC_upper", "KC_middle", "KC_lower"]]
    y = clean_signals['target']

    # add , random_state=42 for a seed
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, test_size=0.5, random_state = 42)
    # Initialize and train the classifier

    clf = GradientBoostingClassifier()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    capital = 10000
    holdings = 0
    last_buy_index = None

    for i in range(len(predictions)):
        if predictions[i] == 1 and capital > 0:
            holdings = capital / df.loc[i, 4]
            capital = 0
            last_buy_index = i

        if last_buy_index is not None and (i - last_buy_index >= 90):
            if holdings > 0:
                capital = holdings * df.loc[i, 4]
                holdings = 0

    # Sell any remaining holdings at the last available price
    if holdings > 0:
        capital = holdings * df[4].iloc[-1]
        holdings = 0

    print("Final capital:", capital)
    
names = ["B6", "AD", "L", "J1", "E7", "E6", "MP", "E1", "A6", "BR", "J7", "RP", "N6", "FBON", "CNH", "T6", "RU", "SEK", "KRW", "NOK"]

for name in names: 
    backtest(name)