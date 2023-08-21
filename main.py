#Import Library Needed.
import streamlit as st
from streamlit_option_menu import option_menu
import requests
import pandas as pd
import subprocess
from PIL import Image
import os
from pathlib import Path
import uuid
import moviepy.editor as moviepy


## Function Request API Recognition via Video.
def analyze_video():
    r = requests.post(f"#########")
    return r

## Function Request API Recognition via Image.
def analyze_image():
    r = requests.post(f"#########")
    return r

## Function Model, returned path selected model 
def selected_model(selected):
    if selected == "Model Deteksi Sampah":
        return "model/model_1.pt"
    elif selected == "Model Deteksi Wajah (1)":
        return "model/model_face_recog_1.pt"
    else :
        return "xxx"

## Function to Convert videos .avi to .mp4
def convert_avi_to_mp4(avi_file_path, output_name):
    clip = moviepy.VideoFileClip(avi_file_path)
    clip.write_videofile(output_name)

## Function check path if not there create.
def check_dir_create(path):
    isExisting = os.path.exists(path)
    if isExisting != True:
        os.mkdir(path)

# Hearder 
st.set_page_config(page_title='Modul YOLO Polban', page_icon='assets/polban_ico.png', layout="centered", initial_sidebar_state="auto", menu_items=None)

# UI Layout
## Sidemenu / Sidebar
with st.sidebar:
    choose = option_menu("Input Type", ["Image", "Video", "Live Video"],
                         icons=['grid fill', 'search heart'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#939ca3"},
        "icon": {"color": "orange", "font-size": "20px"}, 
        "nav-link": {"font-size": "17px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )

model_list = ["Model Deteksi Sampah", "Model Deteksi Wajah (1)", "Model C"]

## Analyize Images.
if choose == "Image":
    st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Modul YOLO V8 Face and Body Recognition</p>', unsafe_allow_html=True)    
    st.subheader("Input type Image.")

    st.caption("Image Face and Body Recognition")
    #with st.form(key='nlpForm'):
    input_data = st.file_uploader("Input Image", type=['png', 'jpeg', 'jpg'])
    if input_data is not None:
        with st.spinner(text='Loading...'):
            #check if dir available (image)
            check_dir_create('/data/images/')
            check_dir_create('/result/images/')
            #generate unique id
            u_id = str(uuid.uuid1())
            st.image(input_data)
            picture = Image.open(input_data)
            picture = picture.save(f'data/images/{u_id}_{input_data.name}')
            source = f'data/images/{u_id}_{input_data.name}'

    model = selected_model(st.selectbox('Select model', model_list))

    submit_button = st.button(label='Analyze')

    # Button Analyize On-click :
    if submit_button and model != "":
        st.info("Results")
        # Predict Function | For addition Status Execution place '.stderr' in last line code below.
        subprocess.run(['yolo', 'task=detect', 'exist_ok=True', 'project=result', 'name=images', 'mode=predict', 'model='+str(model), 'conf=0.5', 'save=True', 'source={}'.format(source)],capture_output=True, universal_newlines=True)
        # Output Img
        result_img = Image.open(f'result/images/{u_id}_{input_data.name}')
        st.image(result_img, caption='Hasil YOLO Detection')

## Analyzing Videos.
elif choose == "Video":
    st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;}
        span[data-baseweb="tag"]{background-color: #95e85a !important;} 
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Modul YOLO V8 Face and Body Recognition</p>', unsafe_allow_html=True)     
    st.subheader("Input type Video.")
    
    st.caption("Video Face and Body Recognition")
    #with st.form(key='nlpForm'):
    input_data = st.file_uploader("Input Video", type=['mp4', 'mkv'])
    if input_data is not None:
        with st.spinner(text='Loading...'):
            #check if dir available (image)
            check_dir_create('/data/videos/')
            check_dir_create('/result/videos/')
            #generate unique id
            u_id = str(uuid.uuid1())
            st.video(input_data)
            with open(os.path.join("data", "videos", u_id+'_'+input_data.name), "wb") as f:
                f.write(input_data.getbuffer())
            source = f'data/videos/{u_id}_{input_data.name}'

    model = selected_model(st.selectbox('Select model', model_list))

    submit_button = st.button(label='Analyze')

    # Button Analyize On-click :
    if submit_button and model != "":
        st.info("Results")
        # Predict Function | For addition Status Execution place '.stderr' in last line code below.
        subprocess.run(['yolo', 'task=detect', 'exist_ok=True', 'project=result', 'name=videos', 'mode=predict', 'model='+str(model), 'conf=0.5', 'save=True', 'source={}'.format(source)],capture_output=True, universal_newlines=True)
        # Remove Extension File, File Result is .avi
        string_path = str(os.path.splitext('result/videos/'+u_id+'_'+input_data.name)[0])
        # Transform Video from .avi to .mp4
        convert_avi_to_mp4(string_path+'.avi', string_path+'.mp4')
        # Output Video
        st.video(string_path + '.mp4')