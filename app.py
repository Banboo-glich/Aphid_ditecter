import streamlit as st
import cv2 as cv
import numpy as np
from PIL import Image
import detect as detect





def train_models():
    detect.train()
    print("[INFO] Training Detection model done!") 
    
    
def main():
    
    st.sidebar.title("Settings")
    st.sidebar.subheader("Parameters")
    st.markdown(
    """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width:300px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width:300px;
            margin-left:-300px;
        }
        </style>
    """,
    unsafe_allow_html=True,
    )
    
    app_mode = st.sidebar.selectbox('Choose the App Mode', ['Introduction', 'Object Detection'])
    
    if app_mode == 'Introduction':
        
        st.title("How old is the pea aphid?")
        image = Image.open('1-G1-1028-13.jpg')
        st.image(image, caption='-pea aphid-')

        st.header('Why this app ?')

        st.text('1.Instantly determining the age of the pea aphid from images can speed up research into development.')
        st.text('2.By applying this web application, accurate monitoring in the field becomes possible. This allows the field conditions to be accurately grasped, contributing to pest control.')

        st.text('This web app assumes a binocular stereo microscope.')


    elif app_mode == "Object Detection":
        
        st.header("Juvenile stage classification of pea aphid using YOLOv8",)
        
        st.sidebar.markdown("----")
        confidence = st.sidebar.slider("Confidence", min_value=0.0, max_value=1.0, value=0.50)
        
        img_file_buffer_detect = st.sidebar.file_uploader("Upload an image", type=['jpg','jpeg', 'png'], key=0)
        DEMO_IMAGE = "1-G1-1028-13.jpg"
        
        if img_file_buffer_detect is not None:
            img = cv.imdecode(np.fromstring(img_file_buffer_detect.read(), np.uint8), 1)
            image = np.array(Image.open(img_file_buffer_detect))
        else:
            img = cv.imread(DEMO_IMAGE)
            image = np.array(Image.open(DEMO_IMAGE))
        st.sidebar.text("Original Image")
        st.sidebar.image(image)
        
        # predict
        detect.predict(img, confidence, st)
        st.balloons()
        
 

if __name__ == "__main__":
    try:
        
        # RUN THE FOLLOWING ONLY IF YOU WANT TO TRAIN MODEL AGAIN 
        # train_models()
        
        main()
    except SystemExit:
        pass
        

