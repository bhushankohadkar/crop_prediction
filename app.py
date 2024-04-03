import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import base64


def load_model(modelfile):
    loaded_model = pickle.load(open(modelfile, 'rb'))
    return loaded_model


def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file, blur_radius=5):
    bin_str = get_base64(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-attachment: fixed;
    }}
    .stApp::before {{
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        bottom: 0;
        left: 0;
        background: inherit;
        filter: blur({blur_radius}px);
    }}
    .stContent {{
        background-color: rgba(255, 255, 255, 0.8); /* Adjust the opacity as needed */
        padding: 20px;
        border-radius: 10px;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)


st.set_page_config(page_title="Crop Recommender", page_icon="🌿", layout='wide', initial_sidebar_state="collapsed")
set_background('image.png', blur_radius=10)  # Adjust blur_radius as needed


def main():
    # title
    html_temp = """
    <div>
    <h1 style="color:Black;text-align:left;"> Crop Recommendation  🌱 </h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # def get_base64(bin_file):
    #     with open(bin_file, 'rb') as f:
    #         data = f.read()
    #     return base64.b64encode(data).decode()

    # def set_background(png_file):
    #     bin_str = get_base64(png_file)
    #     page_bg_img = '''
    #     <style>
    #     .main {
    #         background-image: url("data:image/png;base64,%s");
    #         background-size: cover;
    #         background-attachment: local;

    #     }
    #     </style>
    #     ''' % bin_str
    #     st.markdown(page_bg_img, unsafe_allow_html=True)

    # set_background('image.png')

    col1, col2 = st.columns(2)

    with col1:
        with st.expander(" :red[ℹ️ Information]", expanded=True):
            st.write(""":Black[
            Crop recommendation is one of the most important aspects of precision agriculture. Crop recommendations are based on a number of factors. Precision agriculture seeks to define these criteria on a site-by-site basis in order to address crop selection issues. While the "site-specific" methodology has improved performance, there is still a need to monitor the systems' outcomes.Precision agriculture systems aren't all created equal. 
            However, in agriculture, it is critical that the recommendations made are correct and precise, as errors can result in significant material and capital loss.]

            """)
        '''
        ## How does it work ❓ 
        Complete all the parameters and the machine learning model will predict the most suitable crops to grow in a particular farm based on various parameters
        '''

    with col2:
        st.subheader(" Find out the most suitable crop to grow in your farm 👨‍🌾")
        N = st.number_input("Nitrogen", 1, 10000)
        P = st.number_input("Phosporus", 1, 10000)
        K = st.number_input("Potassium", 1, 10000)
        temp = st.number_input("Temperature", 0.0, 100000.0)
        humidity = st.number_input("Humidity in %", 0.0, 100000.0)
        ph = st.number_input("Ph", 0.0, 100000.0)
        rainfall = st.number_input("Rainfall in mm", 0.0, 100000.0)

        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        if st.button('Predict'):
            loaded_model = load_model('model2.pkl')
            label_encoder = load_model('label.pkl')
            prediction = loaded_model.predict(single_pred)
            original_label = label_encoder.inverse_transform(prediction)
            col1.write('''
		    ## Results 🔍 
		    ''')
            # col1.success(original_label,"are recommended by the A.I for your farm.")
            col1.write(f"{original_label[0]} is recommended by the A.I for your farm.")

        # st.subheader(" Find out the most suitable crop to grow in your farm 👨‍🌾")
        # N = st.number_input("Nitrogen", 1,10000)
        # P = st.number_input("Phosporus", 1,10000)
        # K = st.number_input("Potassium", 1,10000)
        # temp = st.number_input("Temperature",0.0,100000.0)
        # humidity = st.number_input("Humidity in %", 0.0,100000.0)
        # ph = st.number_input("Ph", 0.0,100000.0)
        # rainfall = st.number_input("Rainfall in mm",0.0,100000.0)
        # moisture=st.number_input("Soil moisture",10,100)

        # feature_list = [N, P, K, temp, humidity, ph, rainfall,moisture]
        # single_pred = np.array(feature_list).reshape(1,-1)

        # if st.button('Predict'):
        #     loaded_model = load_model('model2.pkl')
        #     prediction = loaded_model.predict(single_pred)
        #     col1.write('''
        #     ## Results 🔍 
        #     ''')
        #     # Convert the integer prediction to string and then capitalize the first letter
        #     col1.write(f":black[{str(prediction.item()).title()} are recommended by the A.I for your farm.]")

    #   code for html ☘️ 🌾 🌳 👨‍🌾  🍃

    hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    </style>
    """


hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
