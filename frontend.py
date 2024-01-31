import streamlit as st
import calendar
import datetime
import stock_viz as sviz
import stock_ml as sml
import stock_portfolio as sptf

################# SETTING DE L'APPLICATION #################

st.set_page_config(
    page_title="Stock analysis & portfolio management",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# to remove streamlit features from the deployed dashboard
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        <style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

# changer le param decoration.style.right pour couvrir (20) ou découvrir le logo (à partir de 200) 'running'
st.components.v1.html(
    """
    <script>
    // Modify the decoration on top to reuse as a banner

    // Locate elements
    var decoration = window.parent.document.querySelectorAll('[data-testid="stDecoration"]')[0];
    var sidebar = window.parent.document.querySelectorAll('[data-testid="stSidebar"]')[0];

    // Observe sidebar size
    function outputsize() {
        decoration.style.left = `${sidebar.offsetWidth}px`;
    }

    new ResizeObserver(outputsize).observe(sidebar);

    // Adjust sizes
    outputsize();
    decoration.style.height = "5.0rem";
    decoration.style.right = "20px"; 

    // Adjust image decorations
    decoration.style.backgroundImage = "url(https://www.bankrate.com/2022/08/24115227/what-is-market-volatility.jpeg?auto=webp&optimize=high&crop=16:9&width=912)";
    decoration.style.backgroundSize = "contain";
    </script>        
    """,
    width=0,
    height=0,
)

with st.container():
    st.title("Stock analysis & portfolio management")
    st.markdown(
        '<h1 style="font-size:18px"> This dashboard has been designed as a tool to get a better understanding of trends in the financial market and help make educated decisions</h1>',
        unsafe_allow_html=True,
    )
    st.markdown("##")
    st.markdown(
        "The inputs below will be used for the analysis carried out in all the tabs beginning by --Single stock--"
    )
    col1, col2, col3, col4 = st.columns([0.25, 0.25, 0.25, 0.25], gap="medium")
    with col1:
        stock_ticker = st.text_input("Ticker of the asset")
    with col2:
        date_i = st.text_input("Start date (format YYYY-MM-DD)")
    with col3:
        date_f = st.text_input("End date (to today if left empty)", value=None)

    st.markdown(
        "❗*If you encounter any kind of error, please check the spelling of the ticker and try again*❗"
    )
    st.markdown("##")

    with st.sidebar:
        st.markdown(
            "Check the box below to display basic information about the asset whose ticker you've specified"
        )
        basic_info = st.checkbox("Display basic info")
        if basic_info:
            (
                error,
                name,
                industry,
                sector,
                country,
                nb_employee,
                l_cl_pr,
                mkcap,
                pe,
                beta,
                rev,
                ebm,
                fcf,
                evalue,
                curr,
            ) = sviz.get_ticker_info(stock_ticker)
            st.markdown("##")
            st.markdown(f"Basic information - {name}")
            st.markdown("----------------")
            st.markdown(f"Industry:  {industry}")
            st.markdown(f"Sector: {sector}")
            st.markdown(f"Country: {country}")
            st.markdown(f"Number of employee: {nb_employee:,.0f}")
            st.markdown("----------------")
            st.markdown(f"Last closing price:  {l_cl_pr} {curr}")
            st.markdown(f"Market capitalization:  {mkcap/1000000000:,.2f} b{curr}")
            st.markdown(f"PE ratio: {pe:.2f}")
            st.markdown(f"Beta: {beta}")
            st.markdown("----------------")
            st.markdown(f"Total revenue:  {rev/1000000000:,.2f} b{curr}")
            st.markdown(f"Ebitda Margin:  {ebm*100:.2f}%")
            st.markdown(f"FreeCashFlow: {fcf/1000000000:,.2f} b{curr}")
            st.markdown(f"Enterprise Value: {evalue/1000000000:,.2f} b{curr}")

list_tab = [
    "Single stock analysis",
    "Single stock vs. index",
    "Single stock vs. industry comps",
    "Single stock vs. spec stocks",
    "Single stock sentiment analysis",
    "Single stock trend prediction",
    "Single stock last day of the month prediction",
    "Single stock next x day prediction",
    "Market val. insight",
    "Portfolio simulation",
    "Portfolio optimisation",
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
		background-color: #E72A80;
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

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs(
    [s for s in list_tab]
)

with tab1:
    col1, col2, col3 = st.columns([0.33, 0.33, 0.33], gap="medium")
    with col1:
        type_ana = st.selectbox(
            "Type of hist. visualisation", ["Linear", "Logarithmic"]
        )
        st.markdown("#")
        hist_button = st.button("Historical analysis")
    with col2:
        short_t = st.text_input("Short-term moving average (in days)", value=20)
        long_t = st.text_input("Long-term moving average (in days)", value=60)
        type_trend = st.selectbox("Type of trend analysis", ["Simple", "Exponential"])
    with col3:
        boll_mv_av = st.text_input("Bollinger moving average (in days)", value=60)
        nb_std = st.text_input("Number of standard deviation", value=1)
    if hist_button:
        graph1 = sviz.price_hist_analysis(stock_ticker, date_i, date_f, typo=type_ana)
        st.plotly_chart(graph1)
        graph2 = sviz.candlestick_analysis(stock_ticker, date_i, date_f)
        st.plotly_chart(graph2)
        graph3 = sviz.trend_analysis(
            stock_ticker,
            date_i,
            date_f,
            short_term=int(short_t),
            long_term=int(long_t),
            type=type_trend,
        )
        st.plotly_chart(graph3)
        graph4 = sviz.bollinger_analysis(
            stock_ticker, date_i, date_f, mv_av=int(boll_mv_av), std=int(nb_std)
        )
        st.plotly_chart(graph4)

with tab2:
    dico_index = sviz.index_list()
    col1, col2, col3, col4 = st.columns([0.25, 0.25, 0.25, 0.25], gap="medium")
    with col1:
        index = st.selectbox(
            "Select the index for the relative strength analysis", dico_index.values()
        )
        index_comps_button = st.button("RSI analysis")
    if index_comps_button:
        graph_rsi = sviz.strenght_analysis(
            stock_ticker,
            [i for i in dico_index if dico_index[i] == index][0],
            date_i,
            date_f,
        )
        st.plotly_chart(graph_rsi)

with tab3:
    col1, col2, col3, col4 = st.columns([0.25, 0.25, 0.25, 0.25], gap="medium")
    with col1:
        nb_comps = st.text_input(
            "Specify the number of companies to compare against", value=5
        )
        indus_comps_button = st.button("Industry comps analysis")
    if indus_comps_button:
        indus_comps = sviz.comps_industry(stock_ticker, date_i, date_f, nb=nb_comps)
        st.plotly_chart(indus_comps)

with tab4:
    col1, col2, col3 = st.columns([0.50, 0.20, 0.30], gap="medium")
    with col1:
        list_comps = st.text_input(
            "Indicate the tickers of your choice separated by a comma and a space (ex: AAPL, MSFT...)"
        )
    with col2:
        normalisation = st.selectbox("Data normalization", [True, False])
    other_comps_button = st.button("Comparative analysis")
    if other_comps_button:
        choice_comps, plot, df_comps = sviz.comps_choice(
            stock_ticker, list_comps, date_i, date_f, normalisation
        )
        st.plotly_chart(choice_comps)
        cola, colb = st.columns([0.5, 0.5], gap="large")
        with cola:
            st.pyplot(plot.get_figure())
        with colb:
            st.dataframe(df_comps)

with tab5:
    st.markdown(
        f"By pressing the button, you will get a sentiment analysis on the last hundred news articles about the asset {stock_ticker}, as well as a wordcloud"
    )
    sentiment_button = st.button("Sentiment analysis")
    if sentiment_button:
        box, df_out = sml.sentiment_analysis(stock_ticker)
        wrdcld = sml.wordcloud_analysis(stock_ticker)
        col1, col2 = st.columns([0.5, 0.5], gap="medium")
        with col1:
            st.pyplot(box.get_figure())
        with col2:
            st.dataframe(df_out)
        cola, colb, colc = st.columns([0.25, 0.5, 0.25], gap="small")
        with colb:
            st.pyplot(wrdcld.get_figure())

with tab6:
    st.markdown(
        f"By pressing the button, you will get a prediction regarding the movement of the price for the asset {stock_ticker}"
    )
    trend_button = st.button("Trend prediction")
    if trend_button:
        df, new_predictors = sml.prepa_data_for_class(stock_ticker)
        predictions = sml.backtest(df, new_predictors, start=1250, step=125)
        df_profit, user_profit, actual_profit = sml.strategy_profit(
            df, predictions, n=1
        )

        if df_profit.tail(1)["Predictions"].values[0] == 0:
            st.markdown("Donward trend 📉")
        else:
            st.markdown("Up trend 📈")

        if df_profit.tail(1)["Predictions"].values[0] == 0:
            st.markdown(
                f"As of {df_profit.tail(1)['Predictions'].index[0].date()}, according to our trained model, the price of the asset {stock_ticker} will go down."
            )
        else:
            st.markdown(
                f"As of {df_profit.tail(1)['Predictions'].index[0].date()}, according to our trained model, the price of the asset {stock_ticker} will go up. It could be interesting to buy the asset now (closing price {df_profit.tail(1)['Close'].values[0]:.2f})"
            )
        st.markdown("##")
        st.markdown(
            "Simple strategy to implement based on this model: buy the asset when the model estimates that the price will go up, and sell the asset the next day (when the actual price has gone up)."
        )
        st.markdown(
            f"By following this strategy between {df_profit.index.min().date()} and {df_profit.index.max().date()}, the profit generated would have been {user_profit:.2f} {curr} (for one stock purchased)."
        )
        st.markdown(
            f"The maximum profit being {actual_profit:.2f} {curr} (for one stock purchased)."
        )
        st.markdown(
            f"This model makes it possible to take over {user_profit/actual_profit*100:.2f}% of the available profit for this asset."
        )

with tab7:
    eom_button = st.button("Last day of the month prediction")
    if eom_button:
        days, close_prices = sml.prepa_data_for_svr(stock_ticker)
        day_pred, pred = sml.predict_last_day_month(days, close_prices)
        st.markdown(
            f"For the {day_pred[0][0]} of {calendar.month_name[datetime.date.today().month]}, the closing price of {stock_ticker} is estimated at {pred:.2f} {curr}."
        )

with tab8:
    col1, col2, col3 = st.columns([0.33, 0.33, 0.33], gap="medium")
    with col1:
        nb_days = st.text_input("Number of days to predict", value=60)
    nxd_button = st.button("Next x days prediction")
    if nxd_button:
        df, forecast, actual_data, model = sml.predict_next_x_days(
            stock_ticker, nbdays=int(nb_days)
        )
        graph_price = sml.plot_pred_actual(df, actual_data, forecast)
        cola, colb = st.columns([0.45, 0.55], gap="small")
        with cola:
            st.pyplot(graph_price.get_figure())
        with colb:
            st.dataframe(forecast)
        colc, cold, cole = st.columns([0.20, 0.50, 0.30], gap="small")
        with cold:
            st.plotly_chart(model.plot_parameters())

with tab9:
    list_groups = sptf.available_screener()
    col1, col2 = st.columns([0.25, 0.75], gap="medium")
    with col1:
        chosen_group = st.selectbox("Industry to analyse", list_groups)
        valo_button = st.button("Value comparison")
    if valo_button:
        try:
            tickers, dico_tick = sptf.ticks_from_group(chosen_group)
            fig, df_val = sptf.valuation(tickers, dico_tick)
            col1, col2 = st.columns([0.5, 0.5], gap="small")
            with col1:
                st.pyplot(fig.get_figure())
            with col2:
                st.dataframe(df_val)
            df_und_over = sptf.fair_market_value_by_group(chosen_group)
            cola, colb, colc = st.columns([0.1, 0.80, 0.1], gap="small")
            with colb:
                st.dataframe(df_und_over)
        except:
            st.markdown(
                f"❗There are not relevant tickers for {chosen_group}. Please choose another industry❗"
            )


with tab10:
    col1, col2, col3, col4 = st.columns([0.33, 0.33, 0.20, 0.13], gap="medium")
    with col1:
        list_assets = st.text_input(
            "Tickers to be incl. in the portfolio (NFLX, MSFT...)"
        )
        assets_w = st.text_input(
            "Weight of each asset (0.2, 0.15...); equal weights if left empty",
            value=None,
        )
    with col2:
        date_i_ptf = st.text_input(
            "Start date of the portfolio simulation (format YYYY-MM-DD)"
        )
        date_f_ptf = st.text_input(
            "End date of the portfolio simulation (to today if left empty)", value=None
        )
    with col3:
        st.markdown("##")
        button_simulation = st.button("Portfolio analysis")
    if button_simulation:
        fig_assets, remove_asset_list = sptf.portfolio_simulation(
            list_assets, date_i_ptf, date_f_ptf
        )
        cola, colb, colc = st.columns([0.2, 0.55, 0.25], gap="medium")
        with cola:
            st.markdown("###")
            st.markdown(
                '<span style="color:coral"> VOO is a proxy for the S&P500 (to be used as a benchmark)</span>',
                unsafe_allow_html=True,
            )
        with colb:
            st.pyplot(fig_assets.get_figure())
        with colc:
            st.markdown("###")
            st.markdown(
                f"The following assets should be removed from the portfolio (given their return-to-risk ratio compared to the other assets of the portfolio):"
            )
            st.markdown(f'{", ".join(remove_asset_list)}')
            (
                fig_hist,
                cov_matrix_annual,
                port_variance,
                port_volatility,
                port_exp_return,
            ) = sptf.portfolio_kpi(list_assets, assets_w, date_i_ptf, date_f_ptf)
            st.markdown("##")
            st.markdown(f"Expected annual return:  {port_exp_return*100:.2f}%")
            st.markdown(f"Annual volatility (risk):  {port_volatility*100:.2f}%")
            st.markdown(f"Annual variance:  {port_variance*100:.2f}%")

with tab11:
    col1, col2, col3, col4 = st.columns([0.33, 0.33, 0.20, 0.13], gap="medium")
    with col1:
        list_assets_p = st.text_input(
            "Tickers to be incl. in the portfolio (NFLX, AAPL...)"
        )
        invest = st.text_input("Amount to invest")
    with col2:
        date_i_opt = st.text_input(
            "Start date of the portfolio to optimize (format YYYY-MM-DD)"
        )
        date_f_opt = st.text_input(
            "End date of the portfolio to optimize (to today if left empty)", value=None
        )
    with col3:
        st.markdown("##")
        button_optimisation = st.button("Portfolio optimisation")
    if button_optimisation:
        cleaned_weights, ver = sptf.portfolio_optimisation(
            list_assets_p, date_i_opt, date_f_opt
        )
        st.markdown(f"Expected annual return: {ver[0]*100:.2f}%")
        st.markdown(f"Annual volatility: {ver[1]*100:.2f}%")
        st.markdown(f"Sharpe Ratio: {ver[2]*100:.2f}%")
        df_alloc, leftover = sptf.portfolio_allocation(
            list_assets_p,
            date_i_opt,
            date_f_opt,
            cleaned_weights,
            int(invest),
            dico=None,
        )
        st.dataframe(df_alloc)
        st.markdown(f"Funds remaining: {leftover:.2f} {curr}")
