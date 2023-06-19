import streamlit as st
import streamlit as st
import time
import pandas as pd
import datetime
from plotly import graph_objs as go
import plotly.express as px
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Darts
from darts.models import NBEATSModel, NHiTSModel
from darts import TimeSeries

# Sklearn
from sklearn.preprocessing import MinMaxScaler

# Tensorflow
import tensorflow as tf

def get_ticker():
    # Custom expander font size
    st.markdown("<style> div[data-testid='stExpander'] div[role='button'] p {font-size: 20px; font-weight: bold;}</style>", unsafe_allow_html=True)
    
    # Selecting Ticker
    with st.expander("Ticker"):
        with st.container():
            st.markdown("<h1 style='text-align: center; color: #ff4454;'>Select Ticker</h1>", unsafe_allow_html=True)

            selected_ticker = st.selectbox(
                "Placeholder ticker title", 
                (   "",
                    "PT. Astra Agro Lestari Tbk (AALI)", 
                    "PT. Bank Central Asia Tbk (BBCA)", 
                    "PT. Bank Rakyat Indonesia (BBRI)", 
                    "PT. Bumi Resources Tbk (BUMI)", 
                    "PT. Bank Mega Tbk (MEGA)"),
                label_visibility = "collapsed",
                index = 0
                )
            st.subheader("Selected Ticker:   :blue[" + selected_ticker + "]")
            
            # Display current chart?
            is_price = st.checkbox('Display Price Chart', help = "Display dataset's closing price chart from selected ticker")
    
    return selected_ticker, is_price

def get_duration():
    
    # Number of Days to Predict
    with st.expander("Duration"):
        with st.container():
            # Slider
            st.markdown("<h1 style='text-align: center; color: #ff4454;'>Select Number of Days to Predict</h1>", unsafe_allow_html=True)
            st.write("Date starts from 2021/12/30 to 2022/12/30 (only predict business day(s))")
            n_duration = st.slider("Num. of Days", 0, 261, 0, 1, label_visibility = "collapsed")
            
            # Add selection for displaying MA chart for prediction comparison
            is_ma = st.checkbox("Display MA 50, 100, and 200 chart", help = "Display Moving Average chart for 50, 100, and 200 days for comparison")
            
    return n_duration, is_ma

def get_error(y_true, y_prediction):
    mape_prediction = tf.keras.metrics.mean_absolute_percentage_error(y_true, y_prediction)
    return float(mape_prediction)

def get_prediction(ticker_name, n_duration):

    ticker = ticker_name + '_dataset.csv'
    model_name = ticker_name + '_model.pt'

    df = pd.read_csv(ticker)
    close_df = df[['Date', 'Close']]
    volume_df = df[['Date', 'Volume']]
    
    # Fill Missing data from the previous day data for closing price
    close_df = close_df.fillna(method='ffill')
    close_df = pd.DataFrame(close_df)
    
    # Fill missing data from previous day data for volume
    volume_df = volume_df.fillna(method='ffill')
    volume_df = pd.DataFrame(volume_df)

    # Scale data with minmax scaler for closing price
    mms = MinMaxScaler()
    close = mms.fit_transform(np.array(close_df['Close']).reshape(-1, 1))
    close_df['Close'] = close
    
    # Scale data with minmax scaler for volume
    mms_vol = MinMaxScaler()
    volume = mms_vol.fit_transform(np.array(volume_df['Volume']).reshape(-1, 1))
    volume_df['Volume'] = volume
    
    # Make Time Series Object from data
    series = TimeSeries.from_dataframe(close_df, freq = 'B', time_col = 'Date')
    volume_series = TimeSeries.from_dataframe(volume_df, freq = 'B', time_col = 'Date')

    # Because time series object declared frequency is 'B' - Business Days
    # There will be some missing data (because of holiday or other events) from the object
    # Fill missing data from the previous day data
    
    temp_df = series.pd_dataframe()
    temp_df['Close'] = temp_df.fillna(method='ffill')
    
    
    temp_volume_df = volume_series.pd_dataframe()
    temp_volume_df['Volume'] = temp_volume_df.fillna(method='ffill')
    
    # Convert back to Series Object
    series = TimeSeries.from_dataframe(temp_df, freq = 'B')
    volume_series = TimeSeries.from_dataframe(temp_volume_df, freq = 'B')
    
    # Split into train and test -> 5 Tahun Train, 1 Tahun Test

    train_size = math.floor(len(series) * 5/6)
    train, test = series[ : train_size], series[train_size : ]
    train_vol, test_vol = volume_series[ : train_size], volume_series[train_size : ]
    
    # train.plot(label = 'Train')
    # test.plot(label = 'Test')

    # Load model
    nhits_model = NHiTSModel.load(model_name)
    
    # Predict based on train data for n duration
    model_prediction = nhits_model.predict(series = train, past_covariates = volume_series, n = n_duration)

    # Inverse transform the prediction result back
    prediction_df = model_prediction.pd_dataframe()
    inverse_prediction = mms.inverse_transform(prediction_df)
    prediction_df['Close'] = inverse_prediction

    test_df = test[:n_duration].pd_dataframe()
    inverse_test = mms.inverse_transform(test_df)
    test_df['Close'] = inverse_test
    test_df = test_df.rename(columns =  {'Close' : 'Actual'})
    
    # Calculate MAPE score
    score_nhits = get_error(test_df['Actual'], prediction_df['Close'])
    
    return prediction_df, test_df, score_nhits


def get_predict():
    st.markdown("<h1 style='text-align: center; color: #ff4454;'>Time Series Stock Forecasting with NHiTS</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>*Notes: Datasets used in this prediction of each \n ticker is from 2017/01/01 to 2023/01/01 and only predicts business days!</p>", unsafe_allow_html=True)
    
    ticker_name = ""
    is_price_chart = False
    is_ma_selected = False
    is_predict = False
    n_duration = 0
    
    with st.sidebar:
        
        # Get ticker
        ticker_name, is_price_chart = get_ticker()
        
        st.write("")
        
        # Get Duration
        n_duration, is_ma_selected = get_duration()
        
        if ticker_name == "" or n_duration == 0:
            st.markdown("<p style='text-align: center; color: #ff4454;'>Please Select Ticker and Duration Above to Predict</p>", unsafe_allow_html=True)
        else:
            st.markdown("""---""")
            st.markdown("<h1 style='text-align: center; color: #ff4454; margin-top:-30px;'>Configuration</h1>", unsafe_allow_html=True)
            st.write("Currently Selected: :blue[" + ticker_name + "]")
            st.write("Display Price Chart: :blue[" + str(is_price_chart) + "]")
            st.write("Display MA: :blue[" + str(is_ma_selected) + "]")
            st.write("Duration: :blue[" + str(n_duration) + " Business Day(s)]")
            
            col1, col2, col3 = st.columns(3)
            with col2:
                is_predict = st.button('Predict')
            # If Slider is choosen
            if is_predict:
                st.write('Predicting ' + ticker_name + " for " + str(n_duration) + " business days")
                with st.spinner("Predicting..."):
                    time.sleep(3)
                st.success("Success!")
    
    # Display Info And Prediction
    
    if is_predict:
        # Read Selected Data from Dataset in Database
        ticker_alias = ticker_name[-5:-1]
        df_name = ticker_alias.lower() +"_dataset.csv"
        df = pd.read_csv(df_name)
        close_df = df[['Date', 'Close']]
        
        # Display describe from dataset as well
        st.title("Dataset Description")
        st.subheader(ticker_name)
        st.write(df.describe())
        
        st.markdown("""---""")
        
        # If Price Chart is selected
        if is_price_chart:
            with st.container():
                progress_text = "Preparing Price Chart for " + ticker_alias
                my_bar = st.progress(0, text = progress_text)
                for percent_complete in range(5):
                    time.sleep(0.1)
                    my_bar.progress(percent_complete + 1, progress_text)

                my_bar.empty()
                
                # Display Price Chart
                st.write("")
                st.title("Closing Price Chart for :blue[" + ticker_alias + "]")
                
                def create_plot(df):
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x = df['Date'], 
                            y = df['Close'], 
                            name = 'Close ' + ticker_alias
                            ))
                    fig.update_layout(height = 650, width = 1250)
                    fig.update_xaxes(
                        rangeslider_visible=True,
                        rangeselector=dict(
                            buttons=list([
                                dict(count=1, label="1m", step="month", stepmode="backward"),
                                dict(count=6, label="6m", step="month", stepmode="backward"),
                                dict(count=1, label="YTD", step="year", stepmode="todate"),
                                dict(count=1, label="1y", step="year", stepmode="backward"),
                                dict(count=3, label="3y", step="year", stepmode="backward"),
                                dict(count=5, label="5y", step="year", stepmode="backward"),
                                dict(step="all")
                            ])
                        )
                    )
                    fig.update_layout(
                        xaxis_title = "Date", yaxis_title = "Close Price"
                    )
                    fig.update_layout(showlegend=True)
                    st.plotly_chart(fig)
                
                create_plot(close_df)
                
                st.markdown("""---""")
        
        if is_ma_selected:
            with st.container():
                progress_text2 = "Calculating Moving Average"
                my_bar2 = st.progress(0, text = progress_text2)
                for percent_complete in range(5):
                    time.sleep(0.1)
                    my_bar2.progress(percent_complete + 1, progress_text2)

                my_bar2.empty()
                
                # MA 50, 100, 200
                df['MA50'] = df.Close.rolling(50).mean()
                df['MA100'] = df.Close.rolling(100).mean()
                df['MA200'] = df.Close.rolling(200).mean()
                
                # Compare with MA50
                st.title("Close Price vs MA50")
                fig_ma50 = go.Figure()
                fig_ma50.update_layout(height = 650, width = 1250)
                fig_ma50.add_trace(
                    go.Scatter(
                        x = df['Date'], 
                        y = df['Close'], 
                        line = dict(color='#2596be', width=1.3),
                        name = 'Close ' + ticker_alias
                        ))
                fig_ma50.add_trace(
                    go.Scatter(
                        x = df['Date'], 
                        y = df['MA50'], 
                        line = dict(color='orange', width=1.5),
                        name = 'MA 50'
                        ))
                fig_ma50.update_xaxes(
                    rangeslider_visible=True,
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(count=3, label="3y", step="year", stepmode="backward"),
                            dict(count=5, label="5y", step="year", stepmode="backward"),
                            dict(step="all")
                        ])
                    )
                )
                st.plotly_chart(fig_ma50)
                st.markdown("""---""")
                
                # Compare with MA 100
                st.title("Close Price vs MA100")
                fig_ma100 = go.Figure()
                fig_ma100.update_layout(height = 650, width = 1250)
                fig_ma100.add_trace(
                    go.Scatter(
                        x = df['Date'], 
                        y = df['Close'], 
                        line = dict(color='#49be25', width=1.3),
                        name = 'Close ' + ticker_alias
                        ))
                fig_ma100.add_trace(
                    go.Scatter(
                        x = df['Date'], 
                        y = df['MA100'], 
                        line = dict(color='orange', width=1.5),
                        name = 'MA 100'
                        ))
                fig_ma100.update_xaxes(
                    rangeslider_visible=True,
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(count=3, label="3y", step="year", stepmode="backward"),
                            dict(count=5, label="5y", step="year", stepmode="backward"),
                            dict(step="all")
                        ])
                    )
                )
                st.plotly_chart(fig_ma100)
                st.markdown("""---""")
                
                # Compare with MA 200
                st.title("Close Price vs MA200")
                fig_ma200 = go.Figure()
                fig_ma200.update_layout(height = 650, width = 1250)
                fig_ma200.add_trace(
                    go.Scatter(
                        x = df['Date'], 
                        y = df['Close'], 
                        line = dict(color='#9925be', width=1.3),
                        name = 'Close ' + ticker_alias
                        ))
                fig_ma200.add_trace(
                    go.Scatter(
                        x = df['Date'], 
                        y = df['MA200'], 
                        line = dict(color='orange', width=1.5),
                        name = 'MA 200'
                        ))
                fig_ma200.update_xaxes(
                    rangeslider_visible=True,
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(count=3, label="3y", step="year", stepmode="backward"),
                            dict(count=5, label="5y", step="year", stepmode="backward"),
                            dict(step="all")
                        ])
                    )
                )
                st.plotly_chart(fig_ma200)
                st.markdown("""---""")
                
        prediction_df, test_df, mape_score = get_prediction(ticker_alias.lower(), n_duration)
        prediction_df = prediction_df.reset_index()
        test_df = test_df.reset_index() 
        
        # Plot prediction
        st.title("Stock Prediction with NHiTS")
        nhits_fig = go.Figure()
        nhits_fig.update_layout(height = 650, width = 1250)
        nhits_fig.add_trace(
            go.Scatter(
                x = prediction_df['Date'], 
                y = prediction_df['Close'], 
                line = dict(color='#9925be', width=1.3),
                name = 'Prediction'
                ))
        nhits_fig.add_trace(
            go.Scatter(
                x = test_df['Date'], 
                y = test_df['Actual'], 
                line = dict(color='orange', width=1.5),
                name = 'Actual'
                ))
        nhits_fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        st.plotly_chart(nhits_fig)
        
        st.subheader(ticker_alias + " MAPE Score: " + str(mape_score))
        
        # Combine pred and actual then print table
        st.subheader("Actual vs Prediction Table")
        res_df = test_df.copy()
        res_df['Prediction'] = prediction_df['Close']
        res_df.index = res_df.index + 1
        st.dataframe(data = res_df)
        
        st.markdown("""---""")