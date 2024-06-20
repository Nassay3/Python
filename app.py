import streamlit as st
import pandas as pd
import aiohttp
import asyncio
import pytz
from datetime import datetime
import numpy as np

# Fetch symbols from Binance API
async def fetch_symbols():
    url = 'https://api.binance.com/api/v3/ticker/price'
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.json()
            usdt_pairs = [item['symbol'] for item in data if isinstance(item, dict) and 'symbol' in item and item['symbol'].endswith('USDT')]
            filtered_pairs = [symbol for symbol in usdt_pairs if not ('DOWN' in symbol or 'UP' in symbol)]
            return filtered_pairs

# Fetch klines data for each symbol
async def fetch_klines(session, symbol, interval, limit):
    url = f'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    async with session.get(url, params=params) as response:
        data = await response.json()
        return symbol, interval, data

# VWAP Z-score calculation
def vwap_zscore(klines, period):
    if len(klines) < period:
        return None

    closes = np.array([float(k[4]) for k in klines])
    volumes = np.array([float(k[5]) for k in klines])

    mean = np.sum(closes[-period:] * volumes[-period:]) / np.sum(volumes[-period:])
    vwapsd = np.sqrt(np.mean((closes[-period:] - mean) ** 2))

    zvwap = (closes[-1] - mean) / vwapsd
    return zvwap

# Calculate Session Volume Accumulated $
def compute_session_volume(klines):
    session_volume = 0
    volumes = [float(k[5]) for k in klines]
    hlc3s = [(float(k[2]) + float(k[3]) + float(k[4])) / 3 for k in klines]

    session_volumes = []
    for i in range(len(volumes)):
        if i == 0 or klines[i][0] // 86400000 != klines[i-1][0] // 86400000:
            session_volume = hlc3s[i] * volumes[i]
        else:
            session_volume += hlc3s[i] * volumes[i]
        session_volumes.append(session_volume)

    return session_volumes[-1]  # Return the latest session volume accumulated

# Format volume values with M for million and K for thousand
def format_volume(val):
    if val >= 1e6:
        return f"{val / 1e6:.2f}M"
    elif val >= 1e3:
        return f"{val / 1e3:.2f}K"
    else:
        return f"{val:.2f}"

# Process data
def process_data(data):
    processed_data = []
    for symbol, intervals in data.items():
        if '2h' not in intervals or len(intervals['2h']) == 0:
            continue
        timestamp = datetime.now(tz=pytz.utc).strftime('%H:%M:%S')
        trades_per_second = float(intervals['2h'][-1][8]) / (2 * 60 * 60)  # Trades per second for 2-hour interval

        # Calculate zvwap1 for 2h, 15m, and 1m
        zvwap1_2h = vwap_zscore(intervals['2h'], period=576)
        zvwap1_15m = vwap_zscore(intervals['15m'], period=48) if '15m' in intervals else None
        zvwap1_1m = vwap_zscore(intervals['1m'], period=48) if '1m' in intervals else None

        # Calculate session volume accumulated for 2h
        session_volume = compute_session_volume(intervals['2h'])

        processed_data.append({
            'time': timestamp,
            'symbol': symbol,
            'tps': trades_per_second,
            'zvwap1_1m': zvwap1_1m,
            'zvwap1_15m': zvwap1_15m,
            'zvwap1_2h': zvwap1_2h,
            'session': session_volume
        })
    df = pd.DataFrame(processed_data)
    if 'session' in df:
        df['session'] = df['session'].apply(format_volume)
    return df

def determine_color(value):
    if value is None:
        return 'background-color:rgba(128, 128, 128, 0.5)'  # Gray
    if value < 0:
        return 'background-color: rgba(255, 0, 0, 0.5)'  # Red
    if value < 0.875:
        return 'background-color: rgba(255, 165, 0, 0.5)'  # Orange
    if value < 2.5:
        return 'background-color: rgba(0, 255, 0, 0.5)'  # Green
    return ''  # No color

# Main application
async def main():
    st.title('USDT Pairs Real-Time Data')

    # Filter inputs
    tps_min = st.sidebar.number_input('Min TPS', value=0.0, key='tps_min')
    zvwap1_2h_min = st.sidebar.number_input('Min ZVWAP1 2H', value=-5.0, key='zvwap1_2h_min')
    zvwap1_15m_min = st.sidebar.number_input('Min ZVWAP1 15M', value=-5.0, key='zvwap1_15m_min')
    zvwap1_1m_min = st.sidebar.number_input('Min ZVWAP1 1M', value=-5.0, key='zvwap1_1m_min')
    session_min = st.sidebar.number_input('Min Session Volume', value=0.0, key='session_min')

    apply_filters = st.sidebar.button('Apply Filters')

    symbols = await fetch_symbols()

    async with aiohttp.ClientSession() as session:
        tasks = []
        for symbol in symbols:
            tasks.append(fetch_klines(session, symbol, '2h', 576))
            tasks.append(fetch_klines(session, symbol, '15m', 48))
            tasks.append(fetch_klines(session, symbol, '1m', 48))

        klines_responses = await asyncio.gather(*tasks)

    # Organize the fetched data
    klines_data = {}
    for response in klines_responses:
        symbol, interval, data = response
        if symbol not in klines_data:
            klines_data[symbol] = {}
        klines_data[symbol][interval] = data

    df = process_data(klines_data)

    if apply_filters:
        # Apply filters
        df = df[(df['tps'] >= tps_min) & 
                (df['zvwap1_2h'].apply(lambda x: x is not None and x >= zvwap1_2h_min)) & 
                (df['zvwap1_15m'].apply(lambda x: x is not None and x >= zvwap1_15m_min)) & 
                (df['zvwap1_1m'].apply(lambda x: x is not None and x >= zvwap1_1m_min)) & 
                (df['session'].apply(lambda x: float(x.rstrip('MK')) * (1e6 if 'M' in x else (1e3 if 'K' in x else 1))) >= session_min)]

    # Reorder columns
    df = df[['time', 'symbol', 'tps', 'zvwap1_1m', 'zvwap1_15m', 'zvwap1_2h', 'session']]

    # Display sortable table with colors
    df_style = df.style.applymap(determine_color, subset=['zvwap1_1m', 'zvwap1_15m', 'zvwap1_2h'])

    st.dataframe(df_style)

# Run the application with refresh interval
async def run():
    while True:
        await main()
        await asyncio.sleep(60)  # Refresh every 60 seconds

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run())