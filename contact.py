import streamlit as st


def get_contact():
    st.markdown("<h1 style='text-align: center; color: #ff4454;'>Contact</h1>", unsafe_allow_html=True)
    st.write("")
    st.write("This stock prediction web application was made to fulfill thesis requirement, made by")
    
    st.markdown("""
    <style>
    .change_font_size {
        font-size : 30px !important;
        font-weight : bold !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class = "change_font_size">Nathanael Jeffrey - 2201731485</p>', unsafe_allow_html=True)
    st.markdown('<p class = "change_font_size">Computer Science and Mathematics</p>', unsafe_allow_html=True)
    st.markdown('<p class = "change_font_size">Binus University</p>', unsafe_allow_html=True)
    st.markdown('<p class = "change_font_size">Whatsapp: +6281 222 753 753</p>', unsafe_allow_html=True)
