"""Load stock from Yahoo financial."""
import yfinance as yf


def load_stock(ticker_name, period="max"):
    """Load stock function.

    Given a string, ticker_name, load information data frame (df) and
    return it
    """
    ticker_value = yf.Ticker(ticker_name)
    df = ticker_value.history(period=period)
    # df = ticker_value.history(period='max')
    df = df.reset_index()
    return df


# Get a data frame (stock_df) and print it out
if __name__ == "__main__":
    stock_df = load_stock("AAPL")
    print(stock_df)
    print(stock_df.columns)
