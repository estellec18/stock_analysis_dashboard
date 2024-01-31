import pandas as pd
import numpy as np
import math

import yfinance as yf
from yahooquery import Screener

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
import matplotlib.pyplot as plt


#### SINGLE ASSET ANALYSIS ####


def single_asset_return(ticker: str, start_date: str, end_date: str):
    """For a specific asset and given a defined timeframe returns :
    - the simple holding period return (only capital gain)
    - the holding period return (capital gain + dividend yield)
    - the annualized holding period return
    - the variance (measure of volatility)
    - the standard deviatoion

    Args:
        ticker (str): ticker of the asset
        start_date (str): beg of the period
        end_date (str): end of the period

    Returns:
        float: 5 metrics
    """

    df = yf.download(tickers=ticker, start=start_date, end=end_date, rounding=True)
    shpr = (df.iloc[-1, 4] - df.iloc[0, 4]) / df.iloc[0, 4]

    tick = yf.Ticker(ticker)
    div = tick.dividends
    div.index = div.index.tz_convert(None)
    dividends = div[start_date:end_date]
    dsum = dividends.sum()
    hpr = (df.iloc[-1, 4] - df.iloc[0, 4] + dsum) / df.iloc[0, 4]

    daydelta = df.index.max().date() - df.index.min().date()
    y = daydelta.days / 365.25
    an_hrp = (hpr + 1) ** (1 / y) - 1

    df = df[["Adj Close"]]
    df["Returns"] = df["Adj Close"].pct_change(1)
    varc = np.var(df["Returns"])
    std = math.sqrt(varc)

    return shpr, hpr, an_hrp, varc, std


def sharp_ratio(ticker: str, start_date: str, end_date: str):
    """Returns the hsarp ratio of an asset given a specific timeframe

    Args:
        ticker (str): ticker of the asset
        start_date (str): beg of the period
        end_date (str): end of the period

    Returns:
        float: sharp ratio
    """
    df = yf.download(tickers=ticker, start=start_date, end=end_date, rounding=True)
    df["returns"] = df["Adj Close"].pct_change(1)

    # if stocks is in eur, we use 2y germanbond (www.bloomberg.com/markets/rates-bonds/government-bonds/germany)
    # otherwise we use usd 3m T bill
    tick = yf.Ticker(ticker)
    if tick.get_info()["currency"] == "EUR":
        rf_rate = 0.0257 / 252
    else:
        tbil = yf.Ticker("TBIL")
        rf_rate = tbil.get_info()["yield"] / 252

    df["excess_returns"] = df["returns"] - rf_rate
    sharpe_ratio = (np.sqrt(252) * df["excess_returns"].mean()) / df[
        "excess_returns"
    ].std()
    print(f"Sharpe ratio de {sharpe_ratio}")
    return sharpe_ratio


#### PORTFOLIO ANALYSIS ####


def portfolio_return(tickers: str, wgts: str, start_date: str, end_date: str):
    """Return the holding period return of a portfolio over a specified period + the beta of the porfolio

    Args:
        tickers (str): all the assets in a single string separated by ', '
        wgts (str): all the weights in a single string separated by ', '
        start_date (str): beg of the period
        end_date (str): end of the period

    Returns:
        float: return & beta of the portfolio
    """

    # the inpus are given by the user of the dashboard
    # a string is the easiest way
    list_tickers = tickers.split(", ")
    weights = wgts.split(", ")
    list_weights = [eval(i) for i in weights]

    if sum(list_weights) != 1:
        return "La somme des poids doit être égale à 1"

    st_re = []
    dico_beta = {}
    for i in range(len(list_tickers)):
        shpr, hpr, an_hrp, var, std = single_asset_return(
            list_tickers[i], start_date, end_date
        )
        st_re.append(hpr * list_weights[i])
        info = yf.Ticker(list_tickers[i])
        dico_beta[list_tickers[i]] = info.get_info()["beta"]

    beta_port = 0
    for i in range(len(list_tickers)):
        beta_port += list_weights[i] * dico_beta[list_tickers[i]]

    return sum(st_re), beta_port


def portfolio_simulation(stocks: str, start: str, end: str):
    """(1) Plots stocks on a return/risk basis and (2) lists the stocks to remove because of bad risk/return ratio in comparison to the other stocks of the portfolio

    Args:
        stocks (str): all the assets in a single string separated by ', '
        start (str): beg of period
        end (str): end of period
    """

    # the input is necessary a string because inputed by the user of the dashboard
    list_stocks = stocks.split(", ")
    list_stocks.append("VOO")  # addition of 'VOO' as benchmark for S&P500
    dfi = yf.download(
        tickers=" ".join(list_stocks),
        start=start,
        end=end,
        interval="1d",
        rounding=True,
    )
    dfi = dfi["Adj Close"]

    daily_simple_returns = dfi.pct_change(1)
    annual_return = daily_simple_returns.mean() * 252

    annual_risk = daily_simple_returns.std() * math.sqrt(252)
    df2 = pd.DataFrame()
    df2["Expected_annual_return"] = annual_return
    df2["Expected_annual_risk"] = annual_risk
    df2["Company_ticker"] = df2.index
    df2["Ratio"] = df2["Expected_annual_return"] / df2["Expected_annual_risk"]

    fig, ax = plt.subplots(figsize=(6, 6))
    cols = ["royalblue" if x != "VOO" else "coral" for x in df2.index]
    ax.scatter(df2["Expected_annual_risk"], df2["Expected_annual_return"], c=cols)
    ax.set_xlabel("Annual risk")
    ax.set_ylabel("Annual return")
    for idx, row in df2.iterrows():
        ax.annotate(
            row["Company_ticker"],
            (row["Expected_annual_risk"] + 0.01, row["Expected_annual_return"] + 0.001),
            c="black",
        )

    # list of assets that have a lower return but a higher risk than another asset in this data set
    remove_asset_list = []
    for tick in df2["Company_ticker"].values:
        no_better_assets = df2.loc[
            (df2["Expected_annual_return"] > df2["Expected_annual_return"][tick])
            & (df2["Expected_annual_risk"] < df2["Expected_annual_risk"][tick])
        ].empty
        if no_better_assets == False:
            remove_asset_list.append(tick)

    return fig, remove_asset_list


def portfolio_kpi(stocks: str, start: str, end: str, wghts=None):
    """Compute metrics for a given portfolio(tickers + weigths) for a given epriod

    Args:
        stocks (str): all the assets in a single string separated by ', '
        start (str): beg of period
        end (str): enf od period
        wghts (str): all the weights in a single string separated by ', ' (if None, will apply same weights to all assets)
    """

    list_stocks = stocks.split(", ")
    if wghts is None:
        n = 1 / float(len(list_stocks))
        weights = [n] * len(list_stocks)
        weights = np.array(weights)
    else:
        list_weights = wghts.split(", ")
        weights = [eval(i) for i in list_weights]

    dfi = yf.download(
        tickers=" ".join(list_stocks),
        start=start,
        end=end,
        interval="1d",
        rounding=True,
    )
    dfi = dfi["Adj Close"]

    daily_simple_returns = dfi.pct_change(1)
    annual_returns = daily_simple_returns.mean() * 252
    annual_risk = daily_simple_returns.std() * math.sqrt(252)
    df2 = pd.DataFrame()
    df2["Expected_annual_return"] = annual_returns
    df2["Expected_annual_risk"] = annual_risk
    df2["Company_ticker"] = df2.index
    df2["Ratio"] = df2["Expected_annual_return"] / df2["Expected_annual_risk"]

    assets = df2.index
    num_assets = len(assets)
    w = np.array(weights)

    cov_matrix_annual = daily_simple_returns.cov() * 252
    port_variance = np.dot(w.T, np.dot(cov_matrix_annual, w))
    port_volatility = np.sqrt(port_variance)
    port_exp_return = np.sum(w * annual_returns)

    fig, ax = plt.subplots(figsize=(6, 6))
    for c in dfi.columns.values:
        ax.plot(dfi[c], label=c)
    ax.set_title("Portfolio adj. close price history")
    ax.set_ylabel("Adj. closing price")
    ax.legend()

    return fig, cov_matrix_annual, port_variance, port_volatility, port_exp_return


def portfolio_optimisation(stocks: str, start: str, end: str):
    """Given a list of stocks and a period, returns the weight that would lead to the best perf

    Args:
        stocks (str): all the assets in a single string separated by ', '
        start (str): beg of period
        end (str): end of period

    Returns:
        _type_: _description_
    """
    list_stocks = stocks.split(", ")

    df = yf.download(
        tickers=" ".join(list_stocks),
        start=start,
        end=end,
        interval="1d",
        rounding=True,
    )
    df = df[["Adj Close"]]
    df.columns = df.columns.droplevel()

    mu = expected_returns.mean_historical_return(df)
    S = risk_models.sample_cov(df)

    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()  # w to max sharp ratio
    cleaned_weights = ef.clean_weights()  # sets w below seuil to 0

    ver = ef.portfolio_performance(verbose=True)

    return cleaned_weights, ver


def portfolio_allocation(
    stocks: list, start: str, end: str, clean_w: dict, invest_value: int, dico: None
):
    """Given a list of assets and an amount to invest, return the best allocation for max performance

    Args:
        stocks (str): all the assets in a single string separated by ', '
        start (str): beg period
        end (str): end period
        clean_w (dict): weights invested in each asset
        invest_value (int): amount the investor want to invest
        dico (dict): to have the name associated with the ticker
    """
    list_stocks = stocks.split(", ")

    df = yf.download(
        tickers=" ".join(list_stocks),
        start=start,
        end=end,
        interval="1d",
        rounding=True,
    )
    df = df[["Adj Close"]]
    df.columns = df.columns.droplevel()

    latest_prices = get_latest_prices(df)
    weights = clean_w
    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=invest_value)

    allocation, leftover = da.lp_portfolio()
    # print("Discrete allocation ", allocation)
    # print(f"Funds remaining {leftover:.2f}$")

    stock = allocation.keys()
    nb_stock = allocation.values()
    df_alloc = pd.DataFrame()
    df_alloc["Ticker"] = stock
    if dico is not None:
        df_alloc["Name"] = df_alloc["Ticker"].apply(lambda x: dico[x])
    df_alloc["Share_portf"] = df_alloc["Ticker"].map(lambda x: clean_w[x])
    df_alloc["Share_portf"] = df_alloc["Share_portf"] * 100
    df_alloc["Nb_stock"] = nb_stock
    df_alloc["Current_price"] = df_alloc["Ticker"].apply(lambda x: latest_prices[x])
    df_alloc["Cost"] = df_alloc["Nb_stock"] * df_alloc["Current_price"]

    return df_alloc, leftover


#### COMPARISON FROM AGGREGATED GROUPS IN YAHOO FINANCE ####


def available_screener():
    """Returns the list of groups of stocks already made by yahoo finance (to choose from for portfolion creation)

    Returns:
        list: list of groups
    """
    s = Screener()
    return s.available_screeners


def ticks_from_group(group: str):
    """Given a group of assets provided, return the list of tickers corresponding, and a dictionnary with the name of the ticker
    The list of ticker can then be used to create a portfolio with the function portfolio_allocation

    Args:
        group (str): name of the group we are looking for

    """
    s = Screener()
    data = s.get_screeners(group)
    df = pd.DataFrame(data[group]["quotes"])
    tickers = df["symbol"].tolist()

    dico_tick = {}
    for tick in tickers:
        req = yf.Ticker(tick)
        dico_tick[tick] = req.get_info()["shortName"]

    return tickers, dico_tick


def valuation(list_stocks: list, dico_name: dict):
    """Plot each asset of the list in a marketcap/EV basis

    Args:
        list_stocks (list): _description_

    Returns:
        dataframe: dataframe with market cap and enterprise value for comparison purpose
    """

    # Market capitalization is the sum total of all the outstanding shares of a company (stock's current share price X number of shares outstanding)
    # Generally, large-cap companies are well-established successful companies: steady streams of revenue, slower growth but steady, less volatile
    # Appealing small-cap stocks are those  experiencing accelerated growth, but that growth probably will be at the cost of higher risk and price volatility
    # If a company were to be purchased, market capitalization would reflect the cost to acquire the outstanding equity

    # Enterprise value takes into account the debt that the company has taken on
    # EV = market capitalization + outstanding preferred stock + all debt obligations - cash and cash equivalents

    marketcap = [yf.Ticker(tick).get_info()["marketCap"] for tick in list_stocks]
    ev = []
    for tick in list_stocks:
        try:
            ev.append(yf.Ticker(tick).get_info()["enterpriseValue"])
        except:
            ev.append(np.nan)

    new = pd.DataFrame()
    new["asset"] = list_stocks
    new["name"] = new["asset"].map(lambda x: dico_name[x])
    new["market cap"] = marketcap
    new["market cap"] = new["market cap"] / 1000000000
    new["enterprise value"] = ev
    new["enterprise value"] = new["enterprise value"] / 1000000000
    new = new.dropna()
    new = new.sort_values(by="market cap", ascending=False)

    fig, ax = plt.subplots()
    ax.scatter(x=new["market cap"], y=new["enterprise value"])
    ax.set_xlabel("Market Capitalization (in billion)")
    ax.set_ylabel("Enterprise Value (in billion)")
    for idx, row in new.iterrows():
        ax.annotate(
            row["asset"], (row["market cap"], row["enterprise value"]), c="black"
        )

    return fig, new


def fair_market_value_by_group(group: str):
    """A stock is consideted undervalued if fairmarket value > current price

    Args:
        group (str): name of the group we are looking for
    """
    s = Screener()
    data = s.get_screeners(group)
    df = pd.DataFrame(data[group]["quotes"])
    df = df[
        ["symbol", "displayName", "regularMarketPrice", "trailingPE", "epsCurrentYear"]
    ]
    df.rename(
        columns={
            "regularMarketPrice": "Current_price",
            "trailingPE": "PE_ratio",
            "epsCurrentYear": "Earning_per_share",
        },
        inplace=True,
    )
    df = df.set_index("symbol")
    mean_pe = df["PE_ratio"].mean()
    df["Fair_market_value"] = mean_pe * df["Earning_per_share"]
    df["Over_Under_ratio"] = df["Current_price"] / df["Fair_market_value"]
    df["Value_label"] = np.where(
        df["Over_Under_ratio"] < 1, "Under Valued", "Fair or Over Valued"
    )
    df["Value_percentage"] = abs(df["Over_Under_ratio"] - 1) * 100
    df = df.sort_values(by="Fair_market_value", ascending=False)

    return df
