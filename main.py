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
from streamlit_webrtc import webrtc_streamer
import socket
import glob
## Import File
import helper
import shutil
## Function Request API Recognition via Video.
def save_croped_data(device:str, path:str):
    r = requests.post(f"http://127.0.0.1:8000/crop_img/?device={device}&path={path}")
    return r

def get_cropped_data(stat):
    if stat == "all":
        r = requests.get(f"http://127.0.0.1:8000/crop_img/").json()
        return r
    else :
        return 0

## Function Model, returned path selected model 
def selected_model(selected):
    if selected == "Model Deteksi Sampah (3)":
        path = "model/model_2.pt"
        dir_crop = ["plastik", "kaleng", "kaca"]
        return path, dir_crop
    elif selected == "Model Deteksi Wajah (1)":
        path = "model/model_face_recog_1.pt"
        dir_crop = ["face", "body"]
        return path, dir_crop
    elif selected == "Model Deteksi Wajah dan Badan (2)":
        path = "model/model_face_body_recog_1.pt"
        dir_crop = ["face", "body"]
        return path, dir_crop
    else :
        path = "model/model_face_recog_1.pt"
        dir_crop = ["face", "body"]
        return path, dir_crop

## Function to Convert videos .avi to .mp4
def convert_avi_to_mp4(avi_file_path, output_name):
    clip = moviepy.VideoFileClip(avi_file_path)
    clip.write_videofile(output_name)

## Function check path if not there create.
def check_dir_create(path):
    isExisting = os.path.exists(path)
    if isExisting != True:
        os.mkdir(path)

def count_cropped_img(path, num_label):
    data = ""
    counted = 0
    path = remove_ext(path) + '.txt'
    f = open(path, "r")
    for x in f:
        data_line = x.split()
        if int(data_line[0]) == num_label :
            counted = counted + 1

    return counted

def remove_ext(path):
    return str(os.path.splitext(path)[0])

def get_crop_img(path, num_obj):
    data_src = []
    path = remove_ext(path)
    for i in range(0, num_obj):
        if i == 0:
            data_src.append(path + '.jpg')
        else :
            data_src.append(path + str(i+1) + '.jpg')
    return data_src

def save_person_data(device:str, name:str, path:str):
    r = requests.post(f"http://127.0.0.1:8000/save_img/?device={device}&name={name}&path={path}")
    return r

def load_person_images():
    r = requests.get(f"http://127.0.0.1:8000/save_img/")

    return r.json()


# Hearder 
st.set_page_config(page_title='Modul YOLO Polban', page_icon='assets/polban_ico.png', layout="centered", initial_sidebar_state="auto", menu_items=None)

# UI Layout
## Sidemenu / Sidebar
with st.sidebar:
    choose = option_menu("Menu", ["Image Detection", "Video Detection", "Live Video CAM Detection", "Live Video RTSP Detection", 'View Data', 'Add Person', 'View Person'],
                         icons=['grid fill', 'search heart'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#939ca3"},
        "icon": {"color": "orange", "font-size": "20px"}, 
        "nav-link": {"font-size": "17px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )

model_list = ["Model Deteksi Wajah (1)", "Model Deteksi Wajah dan Badan (2)", "Model Deteksi Sampah (3)"]

## Analyize Images.
if choose == "Image Detection":
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
            #generate unique id
            u_id = str(uuid.uuid1())
            st.image(input_data)
            picture = Image.open(input_data)
            picture = picture.save(f'data/images/{u_id}_{input_data.name}')
            source = f'data/images/{u_id}_{input_data.name}'

    model, crop = selected_model(st.selectbox('Select model', model_list))

    submit_button = st.button(label='Analyze')

    # Button Analyize On-click :
    if submit_button and model != "":
        st.info("Results")
        # Predict Function | For addition Status Execution place '.stderr' in last line code below.
        subprocess.run(['yolo', 'task=detect', 'exist_ok=True', 'project=result', 'name=images', 'mode=predict', 'save_txt=True',
            'model='+str(model), 'conf=0.5', 'save=True', 'show=True', 'save_crop=True', 'source={}'.format(source)],
            capture_output=True, universal_newlines=True)
        # Output Img
        result_img = Image.open(f'result/images/{u_id}_{input_data.name}')
        st.image(result_img, caption='Hasil YOLO Detection')
        st.info("Daftar Cropped Image")
        total_cropped = count_cropped_img(f'result/images/labels/{u_id}_{input_data.name}', 0)
        source_crop = get_crop_img(f'result/images/crops/{crop[0]}/{u_id}_{input_data.name}', total_cropped)
        for data_path in source_crop :
            result = save_croped_data(socket.gethostname(), data_path)
            st.image(data_path)
        st.info(result)

## Analyzing Videos.a
elif choose == "Video Detection":
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
            #generate unique id
            u_id = str(uuid.uuid1())
            st.video(input_data)
            with open(os.path.join("data", "videos", u_id+'_'+input_data.name), "wb") as f:
                f.write(input_data.getbuffer())
            source = f'data/videos/{u_id}_{input_data.name}'

    model, crop = selected_model(st.selectbox('Select model', model_list))

    submit_button = st.button(label='Analyze')

    # Button Analyize On-click :
    if submit_button and model != "":
        st.info("Results")
        # Predict Function | For addition Status Execution place '.stderr' in last line code below.
        subprocess.run(['yolo', 'task=detect', 'exist_ok=True', 'project=result', 'name=videos', 'mode=predict', 'model='+str(model), 'conf=0.5', 'save=True', 'save_txt=True', 'source={}'.format(source)],capture_output=True, universal_newlines=True)
        # Remove Extension File, File Result is .avi
        string_path = str(os.path.splitext('result/videos/'+u_id+'_'+input_data.name)[0])
        # Transform Video from .avi to .mp4
        convert_avi_to_mp4(string_path+'.avi', string_path+'.mp4')
        # Output Video
        st.video(string_path + '.mp4')

## Analyzing Videos.
elif choose == "Live Video CAM Detection":
    st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;}
        span[data-baseweb="tag"]{background-color: #95e85a !important;} 
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Modul YOLO V8 Face and Body Recognition</p>', unsafe_allow_html=True)     
    st.subheader("Input type Video.")
    
    st.caption("Video Face and Body Recognition Live CAM")
    #with st.form(key='nlpForm'):
    model, crop = selected_model(st.selectbox('Select model', model_list))

    helper.play_webcam(0.5, helper.load_model(model))

## Analyzing Videos.
elif choose == "Live Video RTSP Detection":
    st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;}
        span[data-baseweb="tag"]{background-color: #95e85a !important;} 
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Modul YOLO V8 Face and Body Recognition</p>', unsafe_allow_html=True)     
    st.subheader("Input type Video.")
    
    st.caption("Video Face and Body Recognition Live RTSP")
    #with st.form(key='nlpForm'):
    model, Cropped = selected_model(st.selectbox('Select model', model_list))

    helper.play_rtsp_stream(0.5, helper.load_model(model))

elif choose == "View Data":
    st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;}
        span[data-baseweb="tag"]{background-color: #95e85a !important;} 
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Database Modul YOLO V8 Face and Body Recognition</p>', unsafe_allow_html=True)     
    st.subheader("Data View.")

    data = get_cropped_data('all')
    for path in data:
        st.write(path)

elif choose == "Add Person":
    st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;}
        span[data-baseweb="tag"]{background-color: #95e85a !important;} 
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Database Modul YOLO V8 Face and Body Recognition</p>', unsafe_allow_html=True)     
    st.subheader("Add Person to Database.")
    input_name = st.text_input("Name", value="person")
    input_data = st.file_uploader("Input Image", type=['png', 'jpeg', 'jpg'])
    if input_data is not None:
        with st.spinner(text='Loading...'):
            #generate unique id
            u_id = str(uuid.uuid1())
            st.image(input_data)
            picture = Image.open(input_data)
            picture = picture.save(f'data/images/{u_id}_{input_data.name}')
            source = f'data/images/{u_id}_{input_data.name}'
    crop =  ""
    submit_button_person = st.button(label='Add User')
    print(submit_button_person)
    if submit_button_person == True:
        st.info("Results")
        # Predict Function | For addition Status Execution place '.stderr' in last line code below.
        subprocess.run(['yolo', 'task=detect', 'exist_ok=True', 'project=result', 'name=images', 'mode=predict', 'save_txt=True',
            'model=model/model_face_recog_1.pt', 'conf=0.5', 'save=True', 'show=True', 'save_crop=True', 'source={}'.format(source)],
            capture_output=True, universal_newlines=True)
        # Output Img
        result_img = Image.open(f'result/images/{u_id}_{input_data.name}')
        st.image(result_img, caption='Hasil YOLO Detection')
        total_cropped = count_cropped_img(f'result/images/labels/{u_id}_{input_data.name}', 0)
        source_crop = get_crop_img(f'result/images/crops/face/{u_id}_{input_data.name}', total_cropped)
        for data_path in source_crop :
            shutil.copy(data_path, 'result/images/person/face/')
            print(data_path)
            result = save_croped_data(socket.gethostname(), data_path)
            st.image(data_path)
            result = save_person_data(socket.gethostname(), input_name, data_path)
        st.info(result)

elif choose == "View Person":
    st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;}
        span[data-baseweb="tag"]{background-color: #95e85a !important;} 
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Database Modul YOLO V8 Face and Body Recognition</p>', unsafe_allow_html=True)     
    st.subheader("List of Registered Person.")
    image_files=load_person_images()
    images = [{"image": item["imgpath"], "name": item["name"]}  for item in image_files]
    #for data_path in image_files :
    print(images)
    for image in images:
        st.text(image['name'])
        st.image(image['image'], width=256)
    # st.text
    # st.image("result/images/person/face/"+image_files)