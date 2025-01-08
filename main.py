
#!pip install streamlit
#!pip install pandas
#!pip install numpy
#!pip install scikit-learn


import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu
import aitool

class MultiApp:

    def __init__(self):
        self.apps = []

    def add_app(self, title, func):

        self.apps.append({
            "title": title,
            "function": func
        })


# ---------------------------
# Streamlit App Layout
# ---------------------------
def main():
   
    st.set_page_config(page_title = "SFSG AI Tool", initial_sidebar_state='expanded', layout="wide")
    st.markdown("""
    <style>
        [data-testid=stSidebar] {
            background-color: white;
        }
    </style>
    """, unsafe_allow_html=True)

    
    with st.sidebar:   
            img = Image.open( "SFSG_Logo.png")
            st.image(img, width =300)
            app = option_menu(
                menu_title='Main Menu',
                options=['About Us', "AI Sample Assesement Tool"],
                icons=['house-fill','person-circle'],
                menu_icon='chat-text-fill',
                default_index=0,
                styles={
                    "container": {"padding": "0!important","background-color":'green'},
                    "icon": {"color": "white", "font-size": "20px"}, 
                    "menu_title":{"background-color": "white"} ,    
                    "nav-link": {"color":"white","font-size": "20px", "text-align": "center", "margin":"1px", "--hover-color": "#8D272B"},
                    "nav-link-selected": {"background-color": "green"},}
            )
        
    if app =="About Us":
            st.title("About Science For Social Good CIC")
            st.info ("\n Vocational training transforms lives and drives economic growth in developing countries.\
                       It equips individuals with the practical skills and knowledge needed to secure employment, start businesses, and contribute to their community’s development.\
                       Through our work, we constantly explore the significance of vocational training in developing countries and its positive impact on individuals, communities, and overall socio-economic progress.\
                       \n Our direction is genetic analysis testing verticals in Agriculture, Human Identification (Forensic DNA), quality and safety testing (environmental, food and water).\
                       To achieve this goal, SSG-CIC aims to establish Vocational Training Hubs (VTHs) across the globe and build local capacity for training and development.\
                       All of this is possible with your support and volunteering.  \n How you can support us?\
                       \n Our fundraising is through 4 main routes:\
                           \n 1. Non Profit Consulting\
                            \n 2. Crowdfunding\
                            \n 3. Volunteer and Work with us\
                            \n 4. Donate used laboratory equipment, IT Gear and Software")

            st.link_button("\n Click here to view more information", "https://ssg-cic.org/", type = "primary")
        
    if app == "AI Sample Assesement Tool":
            aitool.app()
           

#if __name__ == "__main__":
   # main()
