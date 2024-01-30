import streamlit as st
import calendar
import datetime
import stock_viz as sviz
import stock_ml as sml

################# SETTING DE L'APPLICATION #################

st.set_page_config(
    page_title="Stock analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


with st.container():
    st.title("Stock analysis")
    st.markdown(
        "*This dashboard has been designed as a tool to get a better understanding of trends in the financial market and help make educated decisions*"
    )
    st.markdown("##")

    col1, col2, col3, col4 = st.columns([0.25, 0.25, 0.25, 0.25], gap="medium")
    with col1:
        stock = st.text_input('Please provide the ticker of the company')
    with col2:
        date_i = st.text_input('Start date (format YYYY-MM-DD)')
    with col3:
        date_f = st.text_input('End date (to today if left empty)', value=None)

list_tab = [
    "Stock analysis",
    "Stock versus index",
    "Industry comparison",
    "Comparison with other stocks",
    "Sentiment analysis",
    "Portfolio management",
    "Trend prediction",
    "Last Day of the month prediction",
    "Next x day prediction"
]
# style des tabs
st.markdown(
    """
  <style>
    .stTabs [data-baseweb="tab-list"] {
		gap: 2px;
    }
	.stTabs [data-baseweb="tab"] {
		height: 50px;
        white-space: pre-wrap;
		background-color: #F0F2F6;
        margin: 0px 2px 0px 2px;
		border-radius: 4px 4px 0px 0px;
		gap: 1px;
		padding-top: 2px;
		padding-bottom: 2px;
        padding-right: 5px;
        padding-left: 5px;
    }
  </style>
""",
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([s for s in list_tab])


# with Session State, it's possible to have values persist across reruns for those instances when you don't want your variables reinitialized
# cela permet de ne pas avoir à faire une request vers fastapi à chaque fois que l'utilisateur clique sur un bouton
# on conserve les valeurs récupérés lors de la recherche initiale

with tab1:
    col1, col2, col3 = st.columns([0.33, 0.33, 0.33], gap="medium")
    with col1:
        type_ana = st.selectbox("Type of hist. visualisation",['lin', 'log'])
        st.text('Once your choice is made, \nclick the button below')
        hist_button = st.button("Historical analysis")
    with col2:
        short_t = st.text_input('Short-term moving average (days)', value=20)
        long_t = st.text_input('Long-term moving average (days)', value=60)
        type_trend = st.selectbox('Type of trend analysis', ['simple', 'exponential'])
    with col3:
        boll_mv_av = st.text_input('Bollinger moving average', value=60)
        nb_std = st.text_input('Number of standard dev', value=1)
    if hist_button:
        graph1 = sviz.price_hist_analysis(stock, date_i, date_f, typo=type_ana)
        st.plotly_chart(graph1)
        graph2 = sviz.candlestick_analysis(stock, date_i, date_f)
        st.plotly_chart(graph2)
        graph3 = sviz.trend_analysis(stock, date_i, date_f, short_term=int(short_t), long_term=int(long_t), type=type_trend)
        st.plotly_chart(graph3)
        graph4 = sviz.bollinger_analysis(stock, date_i, date_f, mv_av=int(boll_mv_av), std=int(nb_std))
        st.plotly_chart(graph4)

with tab2:
    dico_index = sviz.index_list()
    col1, col2, col3 = st.columns([0.33, 0.33, 0.33], gap="medium")
    with col1:
        index = st.selectbox("Index",dico_index.values())
    with col2:
        st.text('Once your choice is made, click the button below')
        index_comps_button = st.button("RSI analysis")
    if index_comps_button:
        graph_rsi = sviz.strenght_analysis(stock, [i for i in dico_index if dico_index[i]==index][0], date_i, date_f)
        st.plotly_chart(graph_rsi)

with tab3:
    col1, col2, col3 = st.columns([0.33, 0.33, 0.33], gap="medium")
    with col1:
        nb_comps = st.text_input('Number of comparables', value=5)
    with col2:
        st.text('Once your choice is made, click the button below')
        indus_comps_button = st.button("Industry comps analysis")
    if indus_comps_button:
        indus_comps = sviz.comps_industry(stock, date_i, date_f, nb=nb_comps)
        st.plotly_chart(indus_comps)

with tab4:
    col1, col2 = st.columns([0.5, 0.5], gap="large")
    with col1:
        list_comps = st.text_input("Write the tickers separated by a comma and a space (AAPL, MSFT...)")
    with col2:
        normalisation = st.selectbox('Normalized data ?', [True, False])
    other_comps_button = st.button("Comparative analysis")
    if other_comps_button:
        choice_comps, plot, df_comps = sviz.comps_choice(stock, list_comps, date_i, date_f, normalisation)
        st.plotly_chart(choice_comps)
        col3, col4 = st.columns([0.5, 0.5], gap="medium")
        with col3:
            st.pyplot(plot.get_figure())
        with col4:
            st.dataframe(df_comps)

with tab5:
    sentiment_button = st.button("Sentiment analysis")
    if sentiment_button:
        box, df_out = sml.sentiment_analysis(stock)
        wrdcld = sml.wordcloud_analysis(stock)
        col1, col2 = st.columns([0.5, 0.5], gap="medium")
        with col1:
            st.pyplot(box.get_figure())   
        with col2:
            st.dataframe(df_out)
        st.pyplot(wrdcld.get_figure())

with tab7:
    trend_button = st.button("Trend prediction")
    if trend_button:
        df, new_predictors = sml.prepa_data_for_class(stock)
        predictions = sml.backtest(df, new_predictors, start=1250, step=125)
        df_profit, user_profit, actual_profit = sml.strategy_profit(df, predictions, n=1)

        if df_profit.tail(1)['Predictions'].values[0]==0:
            st.markdown("Donward trend 📉")
        else:
            st.markdown("Up trend 📈")

        if df_profit.tail(1)['Predictions'].values[0]==0:
            st.markdown(f"A la date du {df_profit.tail(1)['Predictions'].index[0].date()}, le modèle estime que le prix de l'action va diminuer. Il ne faut donc pas acheter")
        else:
            st.markdown(f"A la data du {df_profit.tail(1)['Predictions'].index[0].date()}, le modèle estime que le prix de l'action va augmenter. Il faut donc acheter (closing price {df_profit.tail(1)['Close'].values[0]:.2f} usd)")
        st.markdown(f"En suivant ce modèle entre {df_profit.index.min().date()} et {df_profit.index.max().date()}, le profit généré aurait été de {user_profit:.2f} $")
        st.markdown(f"Le profit maximal étant de {actual_profit:.2f} $")

with tab8:
    eom_button = st.button("Last day of the month prediction")
    if eom_button:
        days, close_prices = sml.prepa_data_for_svr(stock)
        day_pred, pred = sml.predict_last_day_month(days, close_prices)
        st.markdown(f"For the {day_pred[0][0]} of {calendar.month_name[datetime.date.today().month]}, the predicted closing price is {pred:.2f} usd")

with tab9:
    col1, col2, col3 = st.columns([0.33, 0.33, 0.33], gap="medium")
    with col1:
        nb_days = st.text_input("Number of days to predict", value=60)
    nxd_button = st.button("Next x days prediction")
    cola, colb = st.columns([0.5, 0.5], gap="small")
    if nxd_button:
        df, forecast, actual_data, model = sml.predict_next_x_days(stock, nbdays=int(nb_days))
        graph_price = sml.plot_pred_actual(df, actual_data, forecast)
        with cola:
            st.pyplot(graph_price.get_figure())
        with colb:
            st.plotly_chart(model.plot_parameters())
