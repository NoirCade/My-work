import streamlit as st
import pandas as pd
import numpy as np
import search_stock
import trend
import bugfixer
import QNA
import GAN
from streamlit_option_menu import option_menu
import streamlit.config
import streamlit.components.v1 as html
from  PIL import Image
import numpy as np
import pandas as pd
import io
import os
os.makedirs('./news', exist_ok=True)
os.makedirs('./wordcloud', exist_ok=True)
os.makedirs('./stylecloud', exist_ok=True)
# [theme]
# primaryColor="black"
# backgroundColor="#FFFFFF"
# secondaryBackgroundColor="#F0F2F6"
# textColor="#262730"
# font="sans serif"

# st.set_page_config(
#     page_title="GPT's NEWS",
#     page_icon="🧊",
#     layout="wide",
#     initial_sidebar_state="expanded",
#     # menu_items={
#     #     'Get Help': '도움!',
#     #     'Report a bug': "버그낫다고? 내 알바 아니야~",
#     #     'About': "몰라 내 알바 아니야"
#    # }
# )


with st.sidebar:
    choose = option_menu("GPT's News", ["Trends", "Stock search","Bugfixer",'Q&A'],
                         icons=['house', 'binoculars',"bug","question"],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "7!important", "background-color": "#fafafa"},
        "icon": {"color": "blue", "font-size": "25px"},
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    })
if choose == "Stock search":
    search_stock.run_home()
elif choose == "Trends":
    trend.trend()
elif choose == "Bugfixer":
    bugfixer.bugfixer()
elif choose =="Q&A":
    QNA.QNA()
# elif choose =='GAN':
#     GAN.GAN()

#실행   streamlit run main.py