import streamlit as st
from PIL import Image

def get_homepage():
    st.markdown("<h1 style='text-align: center; color: #ff4454;'>Time Series Stock Forecasting with NHiTS</h1>", unsafe_allow_html=True)
    st.markdown("<style> div[data-testid='stExpander'] div[role='button'] p {font-size: 20px; font-weight: bold;}</style>", unsafe_allow_html=True)
    
    with st.expander("What is Time Series Forecasting?"):
        st.header("Time Series Forecasting")
        st.write("is a prediction/forecasting technique for :blue[**time series data**] (a historical sequence time stamped data) to predict\
            the future values over a period of time.")
        
        # Image TSF
        st.write("")
        tsf_image = Image.open('./Image/time-series-forecasting.png')
        st.image(tsf_image, caption = "Time Series Forecasting Image", width = 600)
        
        st.write("")
        st.write("Image Source: https://www.springboard.com/blog/data-science/time-series-forecasting/")
        
        
    with st.expander("Why is Time Series Forecasting Important?"):
        st.write("The main purpose of a time series forecasting is for :blue[**analysis/identifying**], whether it's to find a patterns, outliers, and even finding an insight from the data variations.")
        st.write("Time Series Forecasting also shows how the data changes over a period of time to help analyst prepare for what possibly happens in the future.")
        
        st.write("")
        
    with st.expander("What are Time Series Components?"):
        st.write("Time Series are divided into 4 components which are: :green[**Trend**], :green[**Seasonal Trend**], :green[**Cyclical Variations**], :green[**Random/Irregular Variations**]")
        st.write("")
        st.subheader(":green[**Trend**]")
        st.write("Trend is a tendency that happens in a long period of time whether the data tends to increase or decrease. It does not mean the data will always increase or decrease throughout the time period, but when being observed as overall, the data will either increase, decrease or stable (stagnant).")
        
        st.write("")
        trend_image = Image.open('./Image/tsf_trend.png')
        st.image(trend_image, caption = "Time Series Trend Components", width = 400)

        st.write("")
        st.subheader(":green[**Seasonal Trend**]")
        st.write("Seasonal trend is the tendencies of data that will make the same pattern in certain period (usually less or equal to a year), these variation/tendencies can happen due to various things, for example, man-made conventions or natural force (season plays a big part in making this seasonal trend happens).")
        
        st.write("")
        seasonality_image = Image.open('./Image/tsf_seasonality.png')
        st.image(seasonality_image, caption = "Time Series Seasonal Components", width = 400)
        
        st.write("")
        st.subheader(":green[**Cyclical Variations**]")
        st.write("Cyclical Variations is almost similar to seasonal trend but on a longer time span, usually more than a year where the data will shows variations/tendencies. One example is a 'Business Cycle', where it is divided into 4 phase cycle which are, prosperity, recession, depression, and recovery.")
        
        st.write("")
        cyclical_image = Image.open('./Image/tsf_cyclical.png')
        st.image(cyclical_image, caption = "Time Series Cyclical Components", width = 400)
        
        st.write("")
        st.subheader(":green[**Random/Irregular Variations**]")
        st.write("Random/Irregular Variations is the last component of a time series which happens to be irregular or random, where these variations are unpredictable and erratic but it can affect time series datas, such as natural disaster and the most current one being COVID-19.")
        
        st.write("")
        irregular_image = Image.open('./Image/tsf_irregular.png')
        st.image(irregular_image, caption = "Time Series Irregular Components", width = 400)
        
        st.write("")
        st.write("Image Source: https://www.analyticsvidhya.com/blog/2023/02/various-techniques-to-detect-and-isolate-time-series-components-using-python/")
        
    with st.expander("What is NHiTS?"):
        st.write("N-HiTS or Neural Hierarchical Interpolation for Time Series Forecasting is a time series forecasting method that improves on the previous N-BEATS method with the aim of improving accuracy while reducing computational cost used especially in doing long horizon forecasting. This is achieved by adding _Multi-Rate Data Sampling_ and _Hierarchical Interpolation_ that do prediction at different rates to achieve better long horizon forecasting.")
        
        st.write("")
        nhits = Image.open('./Image/nhits.png')
        st.image(nhits, caption = "N-HiTS Architecture", width = 1000)
        
        st.write("")
        st.write("Image Source: N-HiTS Paper (https://arxiv.org/pdf/2201.12886.pdf)")
        
    with st.expander("How To Use This Web Application to Predict Stock?"):
        tutorial_list = ['Click on "Predict" on the left sidebar', 
                         'Choose ticker you want to predict (Additionally you can tick display price chart if you want to see price chart from the ticker you have chosen)', 
                         'Set the duration you want to predict (Additionally you can tick MA chart if you want to see Moving Averages from the ticker you have chosen)',
                         'Configuration will be summarized and if it is correct then press "Predict" button',
                         'Wait a couple seconds and the prediction will be shown in the center screen']

        sentence = ''

        for list in tutorial_list:
            sentence += "- " + list + "\n"

        st.markdown(sentence)
        st.write(":red[*Notes: You can not see configuration section or predict button if you have not choose any ticker and set the duration!]")
        