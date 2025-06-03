from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np

app = FastAPI()

# Enable frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/get_stock_data")
def get_stock_data(ticker: str):
    try:
        df = yf.download(ticker, period="30d")
        if df.empty:
            raise HTTPException(status_code=404, detail="No data found for this ticker")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        df['MA'] = df['Close'].rolling(window=9).mean()

        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))

        df = df.dropna()
        if df.empty:
            raise HTTPException(status_code=400, detail="Not enough data to calculate indicators")

        close_val = df['Close'].iloc[-1]
        ma_val = df['MA'].iloc[-1]
        rsi_val = df['RSI'].iloc[-1]

        if pd.isna(close_val) or pd.isna(ma_val) or pd.isna(rsi_val):
            raise HTTPException(status_code=400, detail="Calculated values contain NaN")

        return {
            "ticker": ticker,
            "close": round(float(close_val), 2),
            "ma": round(float(ma_val), 2),
            "rsi": round(float(rsi_val), 2)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.get("/get_prices")
def get_prices(tickers: str):
    try:
        tickers_list = tickers.split(',')
        data = yf.download(tickers=tickers_list, period="1d", interval="1d", group_by="ticker", progress=False)

        results = {}
        for ticker in tickers_list:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    close = data[ticker]['Close'].iloc[-1]
                else:
                    close = data['Close'].iloc[-1]
                results[ticker] = round(float(close), 2)
            except Exception:
                results[ticker] = None

        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch prices: {str(e)}")

@app.get("/")
def root():
    return {"message": "Stock Data API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)






