import streamlit as st
from streamlit_option_menu import option_menu

# Import all pages
import homepage, predict, contact 

st.set_page_config(
    layout = "wide",
    page_title = 'Stock Prediction Web',
    page_icon = 'chart_with_upwards_trend'
    )

with st.sidebar:
    select_page = option_menu(
        menu_title = None,
        menu_icon = None, default_index = 0, orientation = "vertical",
        options = ["Homepage", "Predict",  "Contact", "---"],
        icons = ["house", "graph-up", "person-vcard", 'gear'], 
        styles = {
            "container": {"padding": "0!important", "background-color": "#f0f2f6", "padding-top":"0%!important"},
            "icon": {"color": "black", "font-size": "25px"}, 
            "nav-link": {"font-size": "22px", "text-align": "left", "margin-top" : "5px", "--hover-color": "#D8D8D8"}
        }
    )

def get_selected_page(select_page):
    if select_page == 'Homepage':
        homepage.get_homepage()
    
    elif select_page == 'Predict':
        predict.get_predict()
        
    elif select_page == 'Contact':
        contact.get_contact()
        
get_selected_page(select_page)