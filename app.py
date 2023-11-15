import streamlit as st
import cv2
import numpy as np
from main import heandle
def main():
    uploaded_image = st.file_uploader("Choose an image...", type="jpg")
    # or using your camera
    
    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        img, made, mssv, score, num_fail, num_none = heandle(image)
        st.image(img, caption="Result")
        st.write('Mã đề: ', made)
        st.write('MSSV: ', mssv)
        st.write('Điểm: ', score)
        st.write('Số câu sai: ', num_fail)
        st.write('Số câu không trả lời: ', num_none)
    camera_image = st.camera_input("Or using your camera")
    if camera_image is not None and uploaded_image is None:
        file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        img, made, mssv, score, num_fail, num_none = heandle(image)
        st.image(img, caption="Result")
        st.write('Mã đề: ', made)
        st.write('MSSV: ', mssv)
        st.write('Điểm: ', score)
        st.write('Số câu sai: ', num_fail)
        st.write('Số câu không trả lời: ', num_none)
if __name__ == "__main__":
    main()