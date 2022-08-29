import streamlit as st
import requests
from PIL import Image
from io import BytesIO

SERVER_URL = 'http://stardust-ai.asuscomm.com:9020/inference'

st.set_page_config(page_title="车道线可视化", page_icon="🛣️", layout="wide",)

st.title('车道线可视化')
# st.text('车道线算法可视化算法')

st.sidebar.info('👉🏻可配置参数')
threshold = st.sidebar.slider('请输入置信度阈值', 0.01, 0.99, 0.2)
cut_ratio = st.sidebar.slider('请输入天空高度占比', 0.0, 0.5, 0.4)

file = st.file_uploader('请上传图片', type=['jpg', 'png'])
if file:
    col1, col2 = st.columns(2)
    with col1:
        st.image(file, caption='上传的图片', use_column_width=True)
        r = requests.post(SERVER_URL, files={'image': file}, params={
                          'cut_ratio': cut_ratio, 'threshold': threshold, 'render': 1})
        assert r.status_code == 200
    with col2:
        image = Image.open(BytesIO(r.content))
        st.image(image, caption='车道线可视化', use_column_width=True)

st.header('')
st.header('')
st.header('')
st.header('')
st.header('')
st.image(
    'http://work.startask.net/static/media/logo2.0b1967eb.png',
    caption='Copyright Stardust @2022',
    width=200
)
