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
from deepface import DeepFace
import socket
import glob
## Import File
import helper
import cv2
import shutil
from ultralytics import YOLO
import pandas as pd
import json

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

## Function MODEL
def save_model(name:str, path:str):
    r = requests.post(f"http://127.0.0.1:8000/model/?name={name}&path={path}")
    return r

def get_model(stat):
    if stat == "all":
        r = requests.get(f"http://127.0.0.1:8000/model/").json()
        return r
    else :
        return 0

def get_model_list() :
    json_model = get_model('all')
    model_list = []

    for x in json_model :
        model_list.append(x['name'])

    return model_list

## Function Model, returned path selected model 
def selected_model(selected):
    json_model = get_model('all')
    for x in json_model :
        if x['name'] == selected :
            path = x['path']

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
    choose = option_menu("Menu", 
        [
        "Image Detection", 
        "Video Detection", 
        "Video Detection2", 
        "Live Video CAM Detection", 
        "Live Video RTSP Detection", 
        'View Data', 
        'Add Person', 
        'View Person',
        'Edit Model List'
        ],
                         icons=['grid fill', 'search heart'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#939ca3"},
        "icon": {"color": "orange", "font-size": "20px"}, 
        "nav-link": {"font-size": "17px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )

model_list_deepface = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]

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

    model, crop = selected_model(st.selectbox('Select model', get_model_list()))

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
        st.image(result_img, caption='Hasil YOLO Detection', width=256)
        st.info("Daftar Cropped Image")
        total_cropped = count_cropped_img(f'result/images/labels/{u_id}_{input_data.name}', 0)
        source_crop = get_crop_img(f'result/images/crops/{crop[0]}/{u_id}_{input_data.name}', total_cropped)
        
        num_columns = 3

        # Calculate the number of rows needed based on the number of items and columns
        num_items = len(source_crop)
        num_rows = (num_items + num_columns - 1) // num_columns
        
        # Create a layout with the specified number of columns
        columns = [st.columns(num_columns) for _ in range(num_rows)]

        for i, data_path in enumerate(source_crop):
            result = save_croped_data(socket.gethostname(), data_path)

            print("data path:")
            print(data_path)
            # Face recognition
            dfs = DeepFace.find(
                img_path=data_path,
                db_path="result/images/person/face",
                model_name=model_list_deepface[1],
                enforce_detection=False
            )

            try:
                # Your existing code here
                pathresult = dfs[0]['identity'][0]
                result_name = os.path.dirname(pathresult)
                result_name = result_name.split('/')[-1]
                result_name = result_name.split('\\')[-1]
            except KeyError as e:
                # Handle the exception here, e.g., print an error message
                print(f"An error occurred: {e}")
                result_name = "Tidak terdaftar"

            # Calculate the row and column index for the current item
            row_index = i // num_columns
            col_index = i % num_columns

            # Place content in the appropriate column
            with columns[row_index][col_index]:
                st.text(result_name)
                st.image(data_path, width=128)

## Analyzing Videos.a
elif choose == "Video Detection2":
    st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;}
        span[data-baseweb="tag"]{background-color: #95e85a !important;} 
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Modul YOLO V8 Face and Body Recognition</p>', unsafe_allow_html=True)     
    st.subheader("Input type Video.")
    
    st.caption("Video Face and Body Recognition")
    #with st.form(key='nlpForm'):
    source_vid = st.file_uploader("Input Video", type=['mp4', 'mkv'])
    if source_vid is not None:
        with st.spinner(text='Loading...'):
            #generate unique id
            u_id = str(uuid.uuid1())
            st.video(source_vid)
            with open(os.path.join("data", "videos", u_id+'_'+source_vid.name), "wb") as f:
                f.write(source_vid.getbuffer())
            source = f'data/videos/{u_id}_{source_vid.name}'

    model_path, crop = selected_model(st.selectbox('Select model', get_model_list()))
    
    submit_button = st.button(label='Analyze')
    if submit_button and model_path != "":
        try:
            directory_path = 'result/videos/predict/crops/face/'
            # List all files in the directory
            file_list = os.listdir(directory_path)

            # Iterate through the files and delete each one
            for file_name in file_list:
                file_path = os.path.join(directory_path, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)

            model = YOLO(model_path)
        except Exception as ex:
            st.error(
                f"Unable to load model. Check the specified path: {model_path}")
            st.error(ex)
        with open(str(source), 'rb') as video_file:
            video_bytes = video_file.read()
        if video_bytes:
            #st.video(video_bytes)
            vid_cap = cv2.VideoCapture(
                source)
            st_frame = st.empty()
            print("here")
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    #image = cv2.resize(image, (720, int(720*(9/16))))
                    res = model.predict(image, conf=0.5, save_crop=True, project='result/videos', mode="predict", exist_ok=True)
                    result_tensor = res[0].boxes
                    res_plotted = res[0].plot()
                    st_frame.image(res_plotted,
                                caption='Detected Video',
                                channels="BGR",
                                use_column_width=True
                                )
                else:
                    vid_cap.release()
                    break
            st.info("Daftar Cropped Image")
            source_crop = os.listdir('result/videos/predict/crops/face/')
            num_columns = 3
            data_names = []
            for i, data_path in enumerate(source_crop):
                data_path = f'result/videos/predict/crops/face/{data_path}'
                result = save_croped_data(socket.gethostname(), data_path)

                print("data path:")
                print(data_path)
                # Face recognition
                dfs = DeepFace.find(
                    img_path=data_path,
                    db_path="result/images/person/face",
                    model_name=model_list_deepface[1],
                    enforce_detection=False
                )

                try:
                    # Your existing code here
                    pathresult = dfs[0]['identity'][0]
                    result_name = os.path.dirname(pathresult)
                    result_name = result_name.split('/')[-1]
                    result_name = result_name.split('\\')[-1]
                except KeyError as e:
                    # Handle the exception here, e.g., print an error message
                    print(f"An error occurred: {e}")
                    result_name = "Tidak terdaftar"
                print(f'data pat = {previous_data_path}')
                if result_name == "Tidak terdaftar":
                    data_names.append({"Name": result_name, "Picture": data_path})
                elif all(entry["Name"] != result_name for entry in data_names):
                    # Append the new name and a corresponding placeholder picture
                    data_names.append({"Name": result_name, "Picture": data_path})
                    print(f'append {result_name}')
                else:
                    print(f'dont append {result_name}')
                    continue
            
            # Calculate the number of rows needed based on the number of items and columns
            num_items = len(data_names)
            num_rows = (num_items + num_columns - 1) // num_columns
            
            # Create a layout with the specified number of columns
            columns = [st.columns(num_columns) for _ in range(num_rows)]
            print(data_names)
            print(len(data_names))
            for i in range(len(data_names)):
                # Calculate the row and column index for the current item
                row_index = i // num_columns
                col_index = i % num_columns

                # Place content in the appropriate column
                with columns[row_index][col_index]:
                    st.text(data_names[i]['Name'])
                    st.image(data_names[i]['Picture'], width=128)


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

    model, crop = selected_model(st.selectbox('Select model', get_model_list()))

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
    model, crop = selected_model(st.selectbox('Select model', get_model_list()))

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
    model, Cropped = selected_model(st.selectbox('Select model', get_model_list()))

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
    submit_button_person = st.button(label='Add User (Detect The Face)')
    submit_button_face = st.button(label='Add User (Save Only)')
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
            isExist = os.path.exists(f'result/images/person/face/{input_name}/')
            if not isExist:
                # Create a new directory because it does not exist
                os.makedirs(f'result/images/person/face/{input_name}/')
                print("The new directory is created!")
            shutil.copy(data_path, f'result/images/person/face/{input_name}/')
            print(data_path)
            result = save_croped_data(socket.gethostname(), data_path)
            st.image(data_path)
            result = save_person_data(socket.gethostname(), input_name, f'result/images/person/face/{input_name}/{u_id}_{input_data.name}')
        deepface_rep = "result/images/person/face/representations_facenet.pkl"
        if os.path.exists(deepface_rep):
            try:
                os.remove(deepface_rep)
                print(f"File '{deepface_rep}' has been deleted.")
            except OSError as e:
                print(f"Error deleting the file '{deepface_rep}': {e}")
        else:
            print(f"File '{deepface_rep}' does not exist.")
        st.info(result)

    if submit_button_face == True:
        isExist = os.path.exists(f'result/images/person/face/{input_name}/')
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(f'result/images/person/face/{input_name}/')
            print("The new directory is created!")
        picture = Image.open(input_data)
        picture = picture.save(f'result/images/person/face/{input_name}/{u_id}_{input_data.name}')
        result = save_person_data(socket.gethostname(), input_name, f'result/images/person/face/{input_name}/{u_id}_{input_data.name}')
        deepface_rep = "result/images/person/face/representations_facenet.pkl"
        if os.path.exists(deepface_rep):
            try:
                os.remove(deepface_rep)
                print(f"File '{deepface_rep}' has been deleted.")
            except OSError as e:
                print(f"Error deleting the file '{deepface_rep}': {e}")
        else:
            print(f"File '{deepface_rep}' does not exist.")
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
    num_columns = 3
    num_items = len(images)
    num_rows = (num_items + num_columns - 1) // num_columns
    columns = [st.columns(num_columns) for _ in range(num_rows)]
    # for image in images:
    #     st.text(image['name'])
    #     st.image(image['image'], width=256)
    # Populate the grid with dynamic data (name and picture)
    for i, row in enumerate(columns):
        for j, col in enumerate(row):
            index = i * num_columns + j
            if index < num_items:
                item = images[index]
                col.write(item["name"])
                col.image(item["image"], use_column_width=True)
                
    # st.text
    # st.image("result/images/person/face/"+image_files)

elif choose == "Edit Model List":
    st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;}
        span[data-baseweb="tag"]{background-color: #95e85a !important;} 
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Database Modul YOLO V8 Face and Body Recognition</p>', unsafe_allow_html=True)     
    
    ## Input data Model
    st.subheader("Add Model.")
    input_name = st.text_input("Name", value="Model Face Recognition [Example]")
    input_data = st.file_uploader("Input Model", type=['pt'])
    if input_data is not None:
        with st.spinner(text='Loading...'):
            #generate unique id
            u_id = str(uuid.uuid1())

    ## If Button Clicked
    submit_button_person = st.button(label='Add New Model')
    if submit_button_person == True:
        # Save uploaded file to 'F:/tmp' folder.
        folder = 'model/'
        path_model = Path(folder, f'{u_id}_{input_data.name}')
        with open(path_model, mode='wb') as w:
            w.write(input_data.getvalue())

        if path_model.exists():
            st.success(f'File {input_data.name} is successfully saved!')
            result = save_model(input_name, path_model)
            #st.info(result)
        else :
            st.warning('Upload model is failed', icon="⚠")

    ## Show data Model
    st.subheader("List of Model.")
    data = get_model('all')
    for path in data:
        st.write(path)