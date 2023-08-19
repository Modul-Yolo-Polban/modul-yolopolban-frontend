#Import Library Needed.
import streamlit as st
from streamlit_option_menu import option_menu
import requests
import pandas as pd

## Function Request API Recognition via Video.
def analyze_video():
    r = requests.post(f"#########")
    return r

## Function Request API Recognition via Image.
def analyze_image():
    r = requests.post(f"#########")
    return r


# UI Layout

## Sidemenu / Sidebar
with st.sidebar:
    choose = option_menu("Input Type", ["Image", "Video", "Live Video"],
                         icons=['grid fill', 'search heart'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "20px"}, 
        "nav-link": {"font-size": "17px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )

## Analyize Article for ALL Emiten.

model_list = ["Model A", "Model B", "Model C"]

if choose == "Image":
    st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Modul YOLO V8 Face and Body Recognition</p>', unsafe_allow_html=True)    
    st.subheader("Input type Image.")

    st.caption("Image Face and Body Recognition")
    with st.form(key='nlpForm'):
        input_data = st.file_uploader("Input Video")
        input_model = st.multiselect(
            'Select model',
            model_list)

        submit_button = st.form_submit_button(label='Analyze')

        str_input_model = ' '.join(input_model)

        # Button Analyize On-click :
        if submit_button:
            st.info("Results")
            # Predict Function 
            predict = analyze_image()
            # Output Status Request
            st.write("Hasil Cenah")
            # Output JSON
            st.json("Hasil Cenah")   

## Request Analyize Article for Spesific Emiten.
elif choose == "Video":
    st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;}
        span[data-baseweb="tag"]{background-color: #95e85a !important;} 
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Modul YOLO V8 Face and Body Recognition</p>', unsafe_allow_html=True)     
    st.subheader("Input type Video.")
    
    st.caption("Video Face and Body Recognition")
    with st.form(key='nlpForm'):
        input_data = st.file_uploader("Input Video")
        input_model = st.multiselect(
            'Select model',
            model_list)

        submit_button = st.form_submit_button(label='Analyze')

        str_input_model = ' '.join(input_model)

        # Button Analyize On-click :
        if submit_button:
            st.info("Results")
            # Predict Function 
            predict = analyze_video()
            # Output Status Request
            st.write("Hasil Cenah")
            # Output JSON
            st.json("Hasil Cenah")        

## Request Analyize Article for Spesific Emiten.
elif choose == "Live Video":
    st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;}
        span[data-baseweb="tag"]{background-color: #95e85a !important;} 
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Modul YOLO V8 Face and Body Recognition</p>', unsafe_allow_html=True)     
    st.subheader("Input type Live Video.")
    
    st.caption("Live Video Face and Body Recognition")
    with st.form(key='nlpForm'):
        input_data = st.camera_input("Input Video")
        input_model = st.multiselect(
            'Select model',
            model_list)

        submit_button = st.form_submit_button(label='Analyze')

        str_input_model = ' '.join(input_model)

        # Button Analyize On-click :
        if submit_button:
            st.info("Results")
            # Predict Function 
            predict = analyze_video()
            # Output Status Request
            st.write("Hasil Cenah")
            # Output JSON
            st.json("Hasil Cenah")     