import apimoex
import requests
import pandas as pd

def get_data(tiker = "SBER", start="2023-01-01", end="2023-10-01", interval=24):
    with requests.Session() as session:
        data = apimoex.get_board_candles(session, tiker, start=start, end=end, interval=interval)
        df = pd.DataFrame(data)
        return df