import pandas as pd

import yfinance as yf
from yahooquery import Screener

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


#### Closing price vs. adj closing price ###
# The adj closing price is the closing price after dividend payouts, stock splits, or the issue of additional shares have been taken into account.
# The adj closing price of a company gives you a more accurate indication of the stock's value than the closing price
# it shows you how much investors paid for shares at the end of a trading day
# The closing price is the raw price : the cash value of the last transacted price before the market closes.


#### linear scale vs. log scale analysis ###
# With a log scale, equal vertical distances on the chart correspond to an equal percentage change.
# It is appropriate when the data move through a range of values representing several orders of magnitude.
# A linear scale is better suited for narrow range

#### Candlestick ####
# A daily candlestick shows the market's open, high, low, and close prices for the day. The wide part is called the "real body".
# This real body represents the price range between the open and close of that day's trading : when the body is red, the close was lower than the open (and inversely when green)
# Just above and below the real body are the vertical lines (shadows): they show the high and low prices of that day's trading.
# If the upper shadow on a down candle is short, it indicates that the open on that day was near the day's high.


#### moving averages ####
# Average of closing price over a specified number of periods: it smoothes out short-term price fluctuations and gives a clearer image of market trends
# Simple moving average = weights each price equally
# Exponential moving average = gives the greatest weight to recent prices (and less to older prices)
# 20-days is comonly used because a month contains roughly 20 days // 60-days is commonly used because it represents a quarter of a year
# moving average is less volatile kpi than price
# a security that has been trading down in price will trade below its moving average (and opp.)
# once price begin to move back up toward its moving av line, the line can serve as resistance level


#### Relative strength comparison ####
# Relative Strength Comparison is used to compare a security's performance with a market index
# When the Relative Strength Comparison indicator is moving up, the security’s price is performing better than the base security/index
# When the indicator is moving sideways, both securities prices are rising and falling by the same percentages
# When the indicator is moving down, the security’s price is performing worse than the base security/index in relative terms.


##### STOCK PRICE ANALYSIS ON ITS OWN #####


def get_ticker_info(ticker: str):
    """Returns basic information regarding the company identified by its ticker

    Args:
        ticker (str): ticker of the company of choice
    """

    tick = yf.Ticker(ticker)

    try:
        print(tick.info["longName"])
    except:
        return("This ticker does not existe, please check again")
    print("--------------------------------")
    print("Company Industry : ", tick.info["industry"])
    print("Company Sector : ", tick.info["sector"])
    print("Company Country : ", tick.info["country"])
    print(f"Number of Full-time employees : {tick.info['fullTimeEmployees']:,.0f}")
    print("--------------------------------")
    print(
        f"Last closing price : {tick.info['financialCurrency']} {tick.info['previousClose']}"
    )
    print(
        f"Market capitalization : {tick.info['financialCurrency']} {tick.info['marketCap']/1000000000:,.0f} b"
    )
    print(f"Price Earnings Ratio : {tick.info['trailingPE']:.2f}")
    print("Company Beta : ", tick.info["beta"])
    print("--------------------------------")
    print(
        f"Total revenue : {tick.info['financialCurrency']} {tick.info['totalRevenue']/1000000000:,.0f} b"
    )
    print(f"EBITDA margin : {tick.info['ebitdaMargins']*100:.2f}%")
    print(
        f"Free Cash Flow : {tick.info['financialCurrency']} {tick.info['freeCashflow']/1000000000:,.0f} b"
    )
    print(
        f"Company valuation : {tick.info['financialCurrency']} {tick.info['enterpriseValue']/1000000000:,.0f} b"
    )


def price_hist_analysis(ticker: str, start: str, end=None, typo="lin"):
    """Return one graph showing the closing price evolution in a specific timeframe. The user can choose the scale : linear or logarithmic

    Args:
        ticker (str): ticker of the stock being analysed (ex: 'MSFT')
        start (str): start of the period being analysed (format YYYY-MM-DD).
        end (str, optional): end of the period being analysed (format YYYY-MM-DD). Defaults to None.
        typo (str): scale of the analysis (lin or log)
    """

    tick = yf.Ticker(ticker)
    if end is not None:
        df = yf.download(
            tickers=ticker, start=start, end=end, interval="1d", rounding=True
        )
    else:
        df = yf.download(tickers=ticker, start=start, interval="1d", rounding=True)

    figure = go.Figure(data=[go.Scatter(x=df.index, y=df.Close)])
    figure.update_layout(
        title=f"{tick.info['shortName']} historical closing price",
        width=1100,
        height=600,
    )
    if typo == "log":
        figure.update_yaxes(title_text="log scale", type="log")

    return(figure)


def combo_price_analysis(ticker: str, start: str, end=None):
    """Returns two plotly graphs of the historical price (linear scale + log scale) & one volume graph
    The user either choose the start date and end date (if no end date is chosen, the last available will be chosen)

    Args:
        ticker (str): ticker of the stock being analysed (ex: 'MSFT')
        start (str): start of the period being analysed (format YYYY-MM-DD).
        end (str, optional): end of the period being analysed (format YYYY-MM-DD). Defaults to None.
    """

    tick = yf.Ticker(ticker)
    if end is not None:
        df = yf.download(
            tickers=ticker, start=start, end=end, interval="1d", rounding=True
        )
    else:
        df = yf.download(tickers=ticker, start=start, interval="1d", rounding=True)

    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            f"{ticker} Stock Prices linear scale ({tick.info['financialCurrency']})",
            f"{ticker} Stock Prices logarithmic scale ({tick.info['financialCurrency']})",
            f"{ticker} Volume",
        ),
        vertical_spacing=0.09,
    )

    fig.add_trace(go.Scatter(x=df.index, y=df.Close), row=1, col=1)
    fig.update_yaxes(title_text=f"{tick.info['financialCurrency']}", row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df.Close), row=2, col=1)
    fig.update_yaxes(title_text="log scale", type="log", row=2, col=1)

    fig.add_trace(go.Bar(x=df.index, y=df.Volume), row=3, col=1)
    fig.update_traces(marker_color="blue", row=3, col=1)

    fig.update_layout(height=1000, width=1300, showlegend=False)

    fig.show()


def date_diff(start_date, end_date):
    start_date = start_date.date()
    end_date = end_date.date()
    delta = end_date - start_date
    return delta.days


def sig_drop(df_stock, seuil=0.1):
    perc_change = df_stock["Close"].pct_change()
    sign_price_drop = perc_change[perc_change < -seuil]
    return sign_price_drop


def significant_drops(ticker: str, start: str, seuil: float, end=None):
    tick = yf.Ticker(ticker)
    if end is not None:
        df = yf.download(
            tickers=ticker, start=start, end=end, interval="1d", rounding=True
        )
    else:
        df = yf.download(tickers=ticker, start=start, interval="1d", rounding=True)

    price_drops = sig_drop(df, seuil=seuil)
    nb_days_btw = []
    n = len(price_drops)
    for i in range(0, n - 1):
        date_diff_btw = date_diff(price_drops.index[i], price_drops.index[i + 1])
        nb_days_btw.append(date_diff_btw)

    # on average days btw drop
    av_days_btw_drop = sum(nb_days_btw) / len(nb_days_btw)
    # on average how high is the drop
    av_price_drop = price_drops.mean()

    print(tick.info["longName"])
    print(f"Time period: {df.index.min().date()} to {df.index.max().date()}")
    print("--------------------------------")
    print(
        f"Prices dropped more than {seuil*100:.0f}% {len(price_drops)} times in that specified time period"
    )
    print(
        f"There were on average {av_days_btw_drop:.2f} days between these drops in prices, and the average drop was {av_price_drop*100:.2f}%"
    )
    print(
        f"The biggest drop in price was {price_drops.min()*100:.2f}% and it happened on {price_drops.idxmin().date()}"
    )

    price_hist_analysis(ticker, start, end=end, typo="lin")


def candlestick_analysis(ticker: str, start: str, end=None):
    """Returns a candlestick of the company being analysed for the time frame indicated

    Args:
        ticker (str): ticker of the stock being analysed (ex: 'MSFT')
        start (str): start of the period being analysed (format YYYY-MM-DD).
        end (str, optional): end of the period being analysed (format YYYY-MM-DD). Defaults to None.
    """

    tick = yf.Ticker(ticker)
    if end is not None:
        df = yf.download(
            tickers=ticker, start=start, end=end, interval="1d", rounding=True
        )
    else:
        df = yf.download(tickers=ticker, start=start, interval="1d", rounding=True)

    figure = go.Figure(
        data=[
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
            )
        ]
    )
    figure.update_layout(
        title=f"{tick.info['shortName']} Candlestick Chart",
        width=1100,
        height=600,
        xaxis_rangeslider_visible=True,
    )
    return(figure)


def ts_analysis_per_month(ticker: str, start: str, end=None):
    """Return 2 graphs side by side : a bar plot showing the mean closing price per month & a boxplot showing the variation in closing price per month

    Args:
        ticker (str): ticker of the stock being analysed (ex: 'MSFT')
        start (str): start of the period being analysed (format YYYY-MM-DD).
        end (str, optional): end of the period being analysed (format YYYY-MM-DD). Defaults to None.
    """

    if end is not None:
        df = yf.download(
            tickers=ticker, start=start, end=end, interval="1d", rounding=True
        )
    else:
        df = yf.download(tickers=ticker, start=start, interval="1d", rounding=True)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    df.groupby(df.index.month).Close.describe()["mean"].plot.bar(
        ax=ax[0], color="royalblue"
    )
    ax[0].set_title(f"{ticker} - Mean closing price per month")
    ax[0].tick_params(axis="x", labelrotation=360)
    ax[0].set_xlabel("Month")
    ax[1] = sns.boxplot(
        data=df.assign(Close=df.Close),
        x=df.index.month.rename("Month"),
        y="Close",
        color="royalblue",
    )
    ax[1].set_title(f"{ticker} - Variation of closing prices per month")


def ts_analysis_per_year(ticker: str, start: str, end=None):
    """Returns a boxplot showing closing price variation per year

    Args:
        ticker (str): ticker of the stock being analysed (ex: 'MSFT')
        start (str): start of the period being analysed (format YYYY-MM-DD).
        end (str, optional): end of the period being analysed (format YYYY-MM-DD). Defaults to None.
    """

    if end is not None:
        df = yf.download(
            tickers=ticker, start=start, end=end, interval="1d", rounding=True
        )
    else:
        df = yf.download(tickers=ticker, start=start, interval="1d", rounding=True)

    fig, ax = plt.subplots(figsize=(20, 5))
    sns.boxplot(
        data=df.assign(Close=df.Close),
        x=df.index.year.rename("Year"),
        y="Close",
        color="royalblue",
    )


def trend_analysis(
    ticker: str, start: str, end=None, short_term=20, long_term=60, type="simple"
):
    """_summary_

    Args:
        ticker (str): ticker of the stock being analysed (ex: 'MSFT')
        start (_type_): start of the period being analysed (format YYYY-MM-DD).
        end (_type_, optional): end of the period being analysed (format YYYY-MM-DD). Defaults to None.
        short_term (int): duration in days of moving average short term
        long_term (int): duration in days of moving average long term
        type (str): simple ou exponential

    """

    tick = yf.Ticker(ticker)
    if end is not None:
        df = yf.download(
            tickers=ticker, start=start, end=end, interval="1d", rounding=True
        )
    else:
        df = yf.download(tickers=ticker, start=start, interval="1d", rounding=True)

    df = df[["Close"]]
    if type == "simple":
        df["short_t_mv"] = df["Close"].rolling(short_term).mean()
        df["long_t_mv"] = df["Close"].rolling(long_term).mean()
    else:
        df["short_t_mv"] = df["Close"].ewm(span=short_term).mean()
        df["long_t_mv"] = df["Close"].ewm(span=long_term).mean()

    fig = px.line(
        df,
        x=df.index,
        y="Close",
        labels={ticker},
        title=f"{ticker} Daily Price Chart with {short_term}-day and {long_term}-day {type} moving averages ({tick.info['financialCurrency']})",
    )
    fig.update_traces(line=dict(color="blue", width=1.5))

    fig.add_scatter(
        x=df.index, y=df["short_t_mv"], name=f"{type} MV {short_term}", line_dash="dot"
    )
    fig.add_scatter(
        x=df.index, y=df["long_t_mv"], name=f"{type} MV {long_term}", line_dash="dot"
    )

    fig.update_layout(height=600, width=1100, showlegend=True)

    return(fig)


def bollinger_analysis(ticker: str, start: str, end=None, mv_av=60, std=1):
    """Returns the closing stock price of a ticker and its bollinger bands

    Args:
        ticker (str): ticker du stock
        start (str): _description_. Defaults to None.
        end (str, optional): _description_. Defaults to None.
        mv_av (int): durée du moving average
        std (int): nb de standard deviation

    """

    tick = yf.Ticker(ticker)
    if end is not None:
        df = yf.download(
            tickers=ticker, start=start, end=end, interval="1d", rounding=True
        )
    else:
        df = yf.download(tickers=ticker, start=start, interval="1d", rounding=True)

    df = df[["Close"]]
    df["mv_av"] = df["Close"].rolling(mv_av).mean()
    df["upper"] = df["mv_av"] + std * df.Close.std()
    df["lower"] = df["mv_av"] - std * df.Close.std()

    fig = px.line(
        df,
        x=df.index,
        y="Close",
        labels={ticker},
        title=f"{ticker} Bollinger Bands using {mv_av}-day and {std} standard deviations ({tick.info['financialCurrency']})",
    )
    fig.update_traces(line=dict(color="blue", width=1.5))

    fig.add_scatter(x=df.index, y=df["mv_av"], name=f"{mv_av}-day MV")
    fig.add_scatter(
        x=df.index,
        y=df["upper"],
        name=f"{mv_av}-day MV plus {std} standard dev",
        line_dash="dot",
    )
    fig.add_scatter(
        x=df.index,
        y=df["lower"],
        name=f"{mv_av}-day MV minus {std} standard dev",
        line_dash="dot",
    )

    fig.update_layout(height=600, width=1100, showlegend=True)

    return fig


##### STOCK PRICES COMPARISONS #####


def index_list():
    """Returns a dictionnary with all the available index : tickers and name

    Returns:
        dict: dictionnary ticker:name
    """
    index = pd.read_html("https://finance.yahoo.com/world-indices/")
    df_index = index[0]
    dict_index = df_index.set_index("Symbol")["Name"].to_dict()
    return dict_index


dico_index = index_list()


def index_compar(index1: str, index2: str, index3: str, period: str):
    """Returns 3 graphs side by side showing the evolution of 3 indexes during the time frame indicated

    Args:
        index1 (str): name of the first index
        index2 (str): name of the second index
        index3 (str): name of the third index
        period (str): time frame (ex: '1y' for one year)
    """

    idx1 = yf.Ticker(index1)
    df_idx1 = idx1.history(period=period)
    idx2 = yf.Ticker(index2)
    df_idx2 = idx2.history(period=period)
    idx3 = yf.Ticker(index3)
    df_idx3 = idx3.history(period=period)

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    ax[0].plot(df_idx1["Close"])
    ax[0].set_title(f"{dico_index[index1]}")
    ax[0].set_ylabel(idx1.info["currency"])
    ax[0].tick_params(axis="x", labelrotation=45)
    ax[1].plot(df_idx2["Close"])
    ax[1].set_title(f"{dico_index[index2]}")
    ax[1].set_ylabel(idx2.info["currency"])
    ax[1].tick_params(axis="x", labelrotation=45)
    ax[2].plot(df_idx3["Close"])
    ax[2].set_title(f"{dico_index[index3]}")
    ax[2].set_ylabel(idx3.info["currency"])
    ax[2].tick_params(axis="x", labelrotation=45)

    fig.tight_layout(pad=3.0)
    return fig


def strenght_analysis(ticker: str, index_tick: str, start: str, end=None):
    """Returns a graph showing the relative strengh of the ticker against an index

    Args:
        ticker (str): ticker of the company analysed
        index_tick (str): ticker of the index against which the company is being analysed
        start (str): beg of the period
        end (str, optional): end of the period
    """

    if end is not None:
        df = yf.download(
            tickers=ticker, start=start, end=end, interval="1d", rounding=True
        )
    else:
        df = yf.download(tickers=ticker, start=start, interval="1d", rounding=True)

    idx = yf.Ticker(index_tick)
    df_idx = idx.history(start=df.index.min().date(), end=df.index.max().date())
    df_idx.index = df_idx.index.tz_convert(None)  # naive timezone pour matcher avec df
    df_idx.index = df_idx.index.date  # date sans heure pour matcher avec df

    df["Relative_Strength"] = df["Close"] / df_idx["Close"]

    figure = px.line(
        df,
        x=df.index,
        y="Relative_Strength",
        title=f"Relative strength analysis : {ticker} against {dico_index[index_tick]}",
    )
    figure.update_layout(height=600, width=1100, showlegend=False)
    return figure


def comps_industry(ticker: str, start: str, end=None, nb=5):
    """Graph with company's price and competitors prices

    Args:
        ticker (str): company's ticker
        start (str): beg of the timeframe
        end (str, optional): end of timeframe
        nb (int, optional): nb of competitors. Defaults to 5.
    """

    tick = yf.Ticker(ticker)
    if end is not None:
        df = yf.download(
            tickers=ticker, start=start, end=end, interval="1d", rounding=True
        )
    else:
        df = yf.download(tickers=ticker, start=start, interval="1d", rounding=True)

    # on récupère l'industrie de la société étudiée
    industry = tick.get_info()["industryKey"]
    industry = industry.replace("-", "_")

    # on récupère toutes les sociétés qui sont dans cette industrie
    s = Screener()
    data = s.get_screeners(industry, count=nb)
    df_industry = pd.DataFrame(data[industry]["quotes"])
    dict_comps = dict(zip(df_industry["symbol"], df_industry["shortName"]))

    fig = px.line(
        df,
        x=df.index,
        y="Close",
        labels={ticker},
        title=f"{ticker} - Comparison with other companies from {industry} industry ({tick.info['financialCurrency']})",
    )
    fig.update_traces(line=dict(color="Black", width=2))

    for comps in dict_comps.keys():
        if comps != ticker:
            comp_tick = yf.Ticker(comps)
            df_comp = comp_tick.history(
                start=df.index.min().date(), end=df.index.max().date()
            )
            fig.add_scatter(
                x=df_comp.index,
                y=df_comp["Close"],
                name=dict_comps[comps],
                line_dash="dot",
                line=dict(width=1.5),
            )

    fig.update_layout(height=600, width=1100, showlegend=True)
    return fig


def comps_choice(ticker: str, list_comps: str, start: str, end=None, norm=False):
    """Graph with company's price and other companies price + dataframe with basic price comps

    Args:
        ticker (str): ticker of company being analysed
        list_comps (str): list of tickers against which the company is being analysed
        start (str): beg of timeframe
        end (_type_, optional): end of timeframe. Defaults to None.
        norm (bool): if true, show normalised variation from 100
    """
    
    list_in = list_comps.split(', ')
    string_ticker = ""
    dict_comps = {}
    for tick in list_in:
        string_ticker = string_ticker + tick + " "
        dict_comps[tick] = yf.Ticker(tick).info["shortName"]

    if end is not None:
        df = yf.download(
            tickers=f"{string_ticker + ticker}",
            start=start,
            end=end,
            interval="1d",
            rounding=True,
        )
    else:
        df = yf.download(
            tickers=f"{string_ticker + ticker}",
            start=start,
            interval="1d",
            rounding=True,
        )

    df_close = df["Close"]
    df_describe = df_close.describe()
    df_describe.style.set_properties(
        subset=[f"{ticker}"], **{"background-color": "aqua"}
    )

    plot = sns.heatmap(df_close.corr(), cmap='Reds', annot=True)

    if norm is True:
        df_close = df_close.div(df_close.iloc[0]).mul(100)

    fig = px.line(df_close, x=df_close.index, y=f"{ticker}", labels={f"{ticker}":''})
    fig.update_traces(line=dict(color="Black", width=2))

    for comps in df_close.columns:
        if comps != f"{ticker}":
            fig.add_scatter(
                x=df_close.index,
                y=df_close[comps],
                name=dict_comps[comps],
                line_dash="dot",
                line=dict(width=1.5),
            )
    fig.update_layout(height=600, width=1100, showlegend=True)
    return fig, plot, df_describe.round()


if __name__ == "__main__":
    # execute when the module is not initialized from an import statement
    ts_analysis_per_month("META", start="2023-01-01")
