"""Does the minimum to plot the closing price of two fixed stocks."""

import matplotlib.pyplot as plt
import numpy as np

from stock_load import load_stock


def make_plot(stock_df, field, stock_label):
    """Create the plot."""
    column = getattr(stock_df, field)
    column = np.array(column, dtype="float")
    plt.plot(stock_df["Date"], column, label=stock_label)
    plt.legend()


def do_plot(stock_df, stock_name):
    """Do plot function.

    Use stock_df, as stock data frame read from the web
    """
    make_plot(stock_df, "Close", "Closing price")
    plt.title(stock_name + " Stock Price")
    plt.show()


def do_duo_plot(stock1_df, stock2_df, name1, name2):
    """Take two data frames and graph both."""
    make_plot(stock1_df, "Close", name1)
    make_plot(stock2_df, "Close", name2)

    plt.title(name1 + " vs " + name2)
    plt.show()


def do_highlow_plot(stock_df, stock_name):
    """Do plot of daily highs and lows.

    Use high_price and low_price columns for one stock, which are passed
    through a stock data frame (stock_df)
    """
    make_plot(stock_df, "High", "Daily highs")
    make_plot(stock_df, "Low", "Daily lows")
    plt.title("High/Low Prices for " + stock_name)
    plt.show()


if __name__ == "__main__":

    period = "10y"

    stock_name1 = "MSFT"
    stock1_df = load_stock(stock_name1, period=period)
    # do_plot(stock_df, stock_name)

    stock_name2 = "AAPL"
    stock2_df = load_stock(stock_name2, period=period)
    # do_plot(stock_df, stock_name)

    # do_duo_plot(stock1_df, stock2_df, stock_name1, stock_name2)

    do_highlow_plot(stock1_df, stock_name1)
    do_highlow_plot(stock2_df, stock_name2)
