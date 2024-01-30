import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import calendar
import datetime

import yfinance as yf
from yahooquery import Screener
from finvizfinance.quote import finvizfinance

# NLP
from textblob import TextBlob
import spacy
from wordcloud import WordCloud

# Classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

# Regression
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from neuralprophet import NeuralProphet


#### NLP ####


def sentiment_ana(text: str):
    """compute the polarity of a text : polarity (-1,1)

    Args:
        text (str): text being analysed

    Returns:
        int: polarity
    """
    return TextBlob(text).sentiment.polarity


def outliers_IQR(df: pd.DataFrame):
    """Returns the instance of a dataframe that are outliers

    Args:
        df (DataFrame): dataframe with the data analysed (df['col'])

    Returns:
        DataFrame: only the instances that are outliers
    """
    q3, q1 = q3, q1 = np.percentile(df, [75, 25])
    iqr = q3 - q1
    outliers = df[((df < (q1 - 1.5 * iqr)) | (df > (q3 + 1.5 * iqr)))]
    return outliers


def sentiment_analysis(ticker: str):
    """Returns a boxplot showing the variation of polarity & highlight tthe outliers + dataframe with detail of outliers

    Args:
        ticker (str): ticker of the company analysed

    Returns:
        DataFrame: only the instances that are outliers
    """

    stock = finvizfinance(ticker, verbose=0)
    news = stock.ticker_news()  # dataframe with the last news (max 100)
    news["polarity"] = news["Title"].apply(sentiment_ana)

    fig, axs = plt.subplots(figsize=(3, 8))
    axs = news[["polarity"]].boxplot()
    axs.set_title(f"{ticker}")
    outliers = outliers_IQR(news["polarity"])
    for title in outliers.index:
        axs.text(
            x=1.02,
            y=outliers[title] - 0.02,
            s=news.iloc[title]["Title"],
            fontsize=11,
            color="blue",
        )

    df_outlier = pd.DataFrame(columns=["Date", "Title", "Link", "polarity"])
    outliers = outliers_IQR(news["polarity"])
    i = 0
    for title in outliers.index:
        df_outlier.loc[i] = [
            news.iloc[title]["Date"],
            news.iloc[title]["Title"],
            news.iloc[title]["Link"],
            news.iloc[title]["polarity"],
        ]
        i += 1
    df_outlier.sort_values(by="polarity", ascending=False)

    return fig, df_outlier


nlp = spacy.load("en_core_web_sm")


def lemmatize(text: str):
    """lemmatization of a text

    Args:
        text (str): text to lemmatize

    Returns:
        str: lemmatized text
    """

    text = text.lower()  # tout en minuscule
    doc = nlp(text)
    tokens = [
        token.lemma_
        for token in doc
        if not (token.is_stop or token.is_punct or token.is_digit or len(token) <= 2)
    ]

    return " ".join(tokens)


def wordcloud_analysis(ticker: str):
    """Returns a word cloud of the articles written about the company

    Args:
        ticker (str): ticker of the colmpany being analysed
    """

    stock = finvizfinance(ticker, verbose=0)
    news = stock.ticker_news()
    news["processed_title"] = news["Title"].apply(lambda txt: lemmatize(txt))

    tick = yf.Ticker(ticker)
    to_delete = lemmatize(tick.info["shortName"])
    list_to_delete = to_delete.split(" ")
    list_to_delete.append("stock")
    list_to_delete.append(tick.info["symbol"].lower())

    corpus = " ".join(news["processed_title"].tolist())
    word_list = corpus.split(" ")
    word_list_updated = [w for w in word_list if w not in list_to_delete]

    wordcloud = WordCloud(
        random_state=42,
        normalize_plurals=False,
        width=600,
        height=300,
        max_words=300,
        background_color="white",
        colormap="Set2",
    )
    wordcloud.generate(" ".join(word_list_updated))
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title(
        f"Word cloud des news relatives à {tick.info['shortName']} \n ({(news['Date'].max()-news['Date'].min()).days} derniers jours)",
        fontsize=16,
    )
    plt.axis("off")
    return fig


#### CLASSIFICATION ####

# To implement a price strategy, knowing the direction of the price movement might be more important than the price itself
# Model to predict if tomorrows price will be higher or lower. If the model predict high, I buy, otherwise I don't


def prepa_data_for_class(ticker: str):
    """Prepare the data so as to feed it to the classification model (creates new features)

    Args:
        ticker (str): ticker of the company being analysed

    Returns:
        pd.DataFrame: the data prepared
        list: the list of features to be used by the model
    """

    df = yf.download(tickers=ticker, period="22y")  # 22 ans max
    df["Tomorrow"] = df["Close"].shift(-1)
    df["Target"] = (df["Tomorrow"] > df["Close"]).astype(int)

    horizons = [2, 5, 60, 250, 1000]  # en jours
    new_predictors = []
    for horizon in horizons:
        rolling_av = df.rolling(horizon).mean()

        ratio_col = f"Close_ratio_{horizon}"
        df[ratio_col] = df["Close"] / rolling_av["Close"]

        trend_col = f"Trend_{horizon}"
        df[trend_col] = df.shift(1).rolling(horizon).sum()["Target"]

        new_predictors += [ratio_col, trend_col]

    df = df.dropna()
    return df, new_predictors


def predict(train: pd.DataFrame, test: pd.DataFrame, predictors: list, model):
    """train the model and return the prediction on test data

    Args:
        train (DataFrame): training set
        test (DataFrame): testing set
        predictors (list): list of features
        model (_type_): model to train

    Returns:
        DataFrame: dataframe with target and prediction side by side by date
    """
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:, 1]
    # to make sure that when the model say price will go up, it will actually go up, we increase the threshold (up will be predicted less time, but more secure)
    preds[preds >= 0.6] = 1
    preds[preds < 0.6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined


def backtest(data: pd.DataFrame, predictors: list, start=1250, step=125):
    """Train the model using backtesting. The data train on the first 'start' days, then test on the following 'step' days, and so on...
    Process to take into account the fact that trends in stock price vary a lot

    Args:
        data (DataFrame): data prepared
        predictors (list): list of features
        start (int, optional): number of days on which to train the data. Defaults to 1250 (about 5y)
        step (int, optional): number of days on which to test the data. Defaults to 125 (about 6m)

    Returns:
        DataFrame: dataframe with target and prediction side by side by date
    """

    model = RandomForestClassifier(
        n_estimators=200, min_samples_split=70, random_state=1
    )

    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i : (i + step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)


def strategy_profit(df: pd.DataFrame, predictions: pd.DataFrame, n=1):
    """Compute the profit that would have been made had stock purchases been made following the prediction of the model (buy if 1, otherwise no)

    Args:
        df (DataFrame): data prepared
        predictions (_type_): predictions made by the model
        n (int, optional): number of stocks purchased each time

    Returns:
        DataFrame: detail of the computation
        int: profit made by using strategy
        int: profit max if the model was perfect
    """
    df_profit = predictions.merge(df[["Close"]], left_index=True, right_index=True)
    df_profit["spent"] = np.where(
        df_profit["Predictions"] == 1, df_profit["Close"] * n, 0
    )
    df_profit["spent_true"] = np.where(
        df_profit["Target"] == 1, df_profit["Close"] * n, 0
    )
    df_profit["profit"] = np.where(
        df_profit["Predictions"].shift(1) == 1,
        df_profit["Close"] * n - df_profit["spent"].shift(1),
        0,
    )
    df_profit["profit_true"] = np.where(
        df_profit["Target"].shift(1) == 1,
        df_profit["Close"] * n - df_profit["spent_true"].shift(1),
        0,
    )
    return df_profit, df_profit.profit.sum(), df_profit.profit_true.sum()


#### PRICE PREDICTION - LAST DAY OF THE MONTH ####


def check_day_month():
    """This model can only be used to predict the last day of the month. The function makes sure that we are at the right modement of the month"""

    day_user = datetime.date.today()
    date_tuple = calendar.monthrange(year=day_user.year, month=day_user.month)

    print(f"This model can only be used to predict the last day of the current month")
    if day_user.day < date_tuple[1] - 1:
        print(f"Please us this model on the {date_tuple[1]} or the day before")


def prepa_data_for_svr(ticker: str):
    """Load and prepare the data for the svr model of prediction.

    Args:
        ticker (str): ticker of the company analysed

    Returns:
        list: list of values for feature (day of the month) and target (closing price)
    """
    df = yf.download(tickers=ticker, period="1mo")

    df_days = df.index
    df_close = df.loc[:, "Close"]

    days = []
    for day in df_days:
        days.append([day.to_pydatetime().day])

    close_prices = []
    for close_price in df_close:
        close_prices.append(float(close_price))

    return days, close_prices


def predict_last_day_month(X: list, y: list):
    """Predict the next day of the month based on the feature (days) and the target(close price) provided

    Args:
        X (list): days of the month (1,2,...29)
        y (list): closing price

    Returns:
        list[int]: day of the week predicted
        float: price prediction
    """

    model = SVR()
    param_grid = {
        "kernel": ["rbf"],
        "C": [
            1,
            10,
            100,
            1000,
        ],  # the strength of the regularization is inversely proportional to C. Must be strictly positive. The penalty is a squared l2 penalty.
        "gamma": [
            0.1,
            1,
            10,
            100,
        ],  # degree of the polynomial kernel function.Ignored by all other kernels
    }

    search = GridSearchCV(model, param_grid, scoring="neg_root_mean_squared_error")
    search.fit(X, y)
    rbf_svr = model.set_params(**search.best_params_)
    rbf_svr.fit(X, y)

    day = [[X[-1][0] + 1]]
    return day, rbf_svr.predict(day)[0]


#### PRICE PREDICTION - NEXT X DAYS ####

# RNN is a class of network that can predict the future (up to a point) beacuse it can analyze time series data.
# Once an RNN learns past patterns in the data, it is able to use its knowledge to forecast the future (assuming the patterns still hold in the future).

# NeuralProphet bridges the gap between traditional time-series models and deep learning methods (cf. paper 'NeuralProphet: Explainable Forecasting at Scale' Nov 2021)


def predict_next_x_days(ticker: str, nbdays=60):
    """Prepare data and fit deep learning model

    Args:
        ticker (str): ticker of the company analysed
        nbdays (int, optional): number of days the user wants to predict. Defaults to 60.

    Returns:
        df (DataFrame): historical data loaded and prepared for the model
        forecast (DataFrame) : predictions made by the model for the next 'nbdays'
        actual_data (DataFrame) : historical data learned by the model while training/fitting
        model (deep learning mode) : fitted model
    """

    df = yf.download(tickers=ticker, period="5y")
    df = df[["Close"]]
    df.reset_index(inplace=True)
    df.columns = [
        "ds",
        "y",
    ]  # must have two columns: 'ds' which has the timestamps and 'y' column which contains the observed values of the time series

    model = NeuralProphet()
    model.fit(df)

    future = model.make_future_dataframe(df, periods=nbdays)
    forecast = model.predict(future)
    actual_data = model.predict(df)

    return df, forecast, actual_data, model


def plot_pred_actual(
    df: pd.DataFrame, actual_data: pd.DataFrame, forecast: pd.DataFrame
):
    """Plot the historical data / the data learned by the model / the data forecasted by the model

    Args:
        df (DataFrame): initial data prepared for the model
        actual_data (DataFrame): data learned by the model
        forecast (DataFrame): data forecasted by the model
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["ds"], df["y"], label="historical_data", c="grey")
    ax.plot(
        actual_data["ds"],
        actual_data["yhat1"],
        label="actual data learned by model",
        c="royalblue",
    )
    ax.plot(forecast["ds"], forecast["yhat1"], label="future prediction", c="red")
    ax.legend()
    ax.set_title("Historical price and prediction")
    return fig

#model.plot_parameters()
