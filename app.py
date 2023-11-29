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
    
    app_mode = st.sidebar.selectbox('Choose the App Mode', ['About App', 'Object Detection'])
    
    if app_mode == 'About App':
        
        st.title("How old is the pea aphid?")
        image = Image.open('1-G1-1028-13.jpg')
        st.image(image, caption='-pea aphid-')

        
        # st.markdown("<style> p{margin: 10px auto; text-align: justify; font-size:20px;}</style>", unsafe_allow_html=True)      
        # st.markdown("<p>🚀Welcome to the introduction page of our project! In this project, we will be exploring the YOLO (You Only Look Once) algorithm. YOLO is known for its ability to detect objects in an image in a single pass, making it a highly efficient and accurate object detection algorithm.🎯</p>", unsafe_allow_html=True)  
        # st.markdown("<p>The latest version of YOLO, YOLOv8, released in January 2023 by Ultralytics, has introduced several modifications that have further improved its performance. 🌟</p>", unsafe_allow_html=True)
        # st.markdown("""<p>🔍Some of these modifications are:<br>
        #             &#x2022; Introducing a new backbone network, Darknet-53,<br>
        #             &#x2022; Introducing a new anchor-free detection head. This means it predicts directly the center of an object instead of the offset from a known anchor box.<br>
        #             &#x2022; and a new loss function.<br></p>""", unsafe_allow_html=True)
        
        # st.markdown("""<p>🎊One of the key advantages of YOLOv8 is its versatility. It not only supports object detection but also offers out-of-the-box support for classification and segmentation tasks. This makes it a powerful tool for various computer vision applications.<br><br>
        #             ✨In this project, we will focus on three major computer vision tasks that YOLOv8 can be used for: <b>classification</b>, <b>detection</b>, and <b>segmentation</b>. We will explore how YOLOv8 can be applied in the field of medical imaging to detect and classify various anomalies and diseases🧪💊.</p>""", unsafe_allow_html=True)
        
        # st.markdown("""<p>We hope you find this project informative and inspiring.💡 Let's dive into the world of YOLOv8 and discover how easy it is to use it!🥁🎆</p>""", unsafe_allow_html=True)
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
        

