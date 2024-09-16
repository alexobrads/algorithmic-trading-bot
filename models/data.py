import os
import os.path
from datetime import datetime, timedelta
import pandas_ta as ta
import pandas as pd
from binance import Client


class HistoricalBars:
    def __init__(self, symbol="ETHAUD", bar="1h", look_back=1000):
        self.symbol = symbol
        self.bar = bar
        self.look_back = look_back
        self.raw_data = self.get_raw_data()
        self.ta_data = pd.DataFrame()
        self.get_ta_data()

    def get_raw_data(self):
        start = datetime.utcnow() - timedelta(days=self.look_back)
        end = datetime.utcnow()

        client = Client()
        bars = client.get_historical_klines(
            self.symbol,
            self.bar,
            start_str=start.strftime("%Y-%m-%d %H:%M:%S"),
            end_str=end.strftime("%Y-%m-%d %H:%M:%S"))

        column_names = [
            'openTime',
            'Open',
            'High',
            'Low',
            'Close',
            'Volume',
            'closeTime',
            'quoteAssetVolume',
            'numberOfTrades',
            'takerBuyBaseVol',
            'takerBuyQuoteVol',
            'ignore']
        df = pd.DataFrame(bars, columns=column_names)
        df = df[['openTime', 'closeTime', 'Open', 'Close', 'High', 'Low', 'Volume']]
        df.closeTime = df.closeTime.astype("float64")
        df.openTime = df.openTime.astype("float64")
        df.Open = df.Open.astype("float64")
        df.Close = df.Close.astype("float64")
        df.High = df.High.astype("float64")
        df.Low = df.Low.astype("float64")
        df.Volume = df.Volume.astype("float64")
        return df

    def get_ta_data(self, strategy="All"):
        self.ta_data = self.raw_data.set_index(pd.DatetimeIndex(self.raw_data["closeTime"]), inplace=False)
        self.ta_data.ta.strategy(strategy)

    def save_raw_data(self):
        os.makedirs("data", exist_ok=True)
        self.raw_data.to_csv(f"data/{self.symbol}_raw.csv")

    def save_data(self):
        os.makedirs("data", exist_ok=True)
        self.ta_data.to_csv(f"data/{self.symbol}_ta.csv")
