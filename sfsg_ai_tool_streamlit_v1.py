
#!pip install streamlit
#!pip install pandas
#!pip install numpy
#!pip install scikit-learn


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from PIL import Image
from streamlit_option_menu import option_menu



# ---------------------------
# Utility Functions
# ---------------------------
def load_csv(file_path):
    """Loads a CSV file and handles errors."""
    try:
        df = pd.read_csv(file_path)
        st.write("Dataset loaded successfully:")
        st.write(df.head())
        return df
    except FileNotFoundError:
        st.error(f"Error: '{file_path}' not found.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

def check_columns(df, required_columns):
    """Ensures required columns exist in the DataFrame."""
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
        st.stop()

def classify_degradation(index):
    """Classify degradation based on the index."""
    if index < 1:
        return "Ready"
    elif 1 <= index < 10:
        return "Degraded"
    else:
        return "Significant Degradation"

# ---------------------------
# Data Preparation
# ---------------------------
def prepare_data(df):
    """Prepare dataset by cleaning and calculating the degradation index."""
    df.columns = df.columns.str.strip()

    required_columns = ['Sample_ID', 'Target_Name', 'Quantity']
    check_columns(df, required_columns)

    autosom_1 = df[df['Target_Name'] == 'Autosom 1'][['Sample_ID', 'Quantity']].rename(columns={'Quantity': 'Quantity_Autosom_1'})
    autosom_2 = df[df['Target_Name'] == 'Autosom 2'][['Sample_ID', 'Quantity']].rename(columns={'Quantity': 'Quantity_Autosom_2'})

    merged_df = pd.merge(autosom_1, autosom_2, on='Sample_ID', how='inner')
    merged_df['Quantity_Autosom_1'] = pd.to_numeric(merged_df['Quantity_Autosom_1'], errors='coerce')
    merged_df['Quantity_Autosom_2'] = pd.to_numeric(merged_df['Quantity_Autosom_2'], errors='coerce')

    merged_df['Degradation_Index'] = merged_df['Quantity_Autosom_2'] / merged_df['Quantity_Autosom_1']
    merged_df = merged_df.drop_duplicates(subset='Sample_ID', keep='first')

    return merged_df

# -----------------------
# Decision Making
# -----------------------

def assess_sample(df, sample_id):
    """Assess the degradation status of a specific sample."""
    sample_data = df[df['Sample_ID'] == sample_id]

    if sample_data.empty:
        st.error(f"Sample ID '{sample_id}' not found in the dataset.")
    else:
        degradation_index = sample_data['Degradation_Index'].iloc[0]
        st.write(f"Sample ID: {sample_id}")
        st.write(f"Degradation Index: {degradation_index}")

        if degradation_index < 1:
            st.success("The sample is not degraded and is ready for further analysis.")
        elif 1 <= degradation_index < 10:
            st.warning("The sample is degraded. No action needed.")
        else:
            st.error("The sample is significantly degraded. Resample is required.")

# ---------------------------
# Machine Learning Model
# ---------------------------
def train_model(df):
    """Train and evaluate a RandomForestClassifier."""
    df['Degradation_Status'] = df['Degradation_Index'].apply(classify_degradation)
    status_mapping = {"Ready": 0, "Degraded": 1, "Significant Degradation": 2}
    df['Degradation_Status'] = df['Degradation_Status'].map(status_mapping)

    X = df[['Quantity_Autosom_1', 'Quantity_Autosom_2', 'Degradation_Index']]
    y = df['Degradation_Status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    st.write("### Model Evaluation")
    st.text(classification_report(y_test, y_pred))
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# ---------------------------
# Streamlit App Layout
# ---------------------------
def main():
   
    st.set_page_config(page_title = "SFSG AI Tool", initial_sidebar_state='expanded')
    st.markdown("""
    <style>
        [data-testid=stSidebar] {
            background-color: white;
        }
    </style>
    """, unsafe_allow_html=True)
    
    df = pd.read_csv("SFSG_Dataset.csv")
    with st.sidebar:   
            img = Image.open( "SFSG_Logo.png")
            st.image(img, width =250)
            app = option_menu(
                menu_title='Main Menu',
                options=['About Us', "AI Sample Assesement Tool"],
                icons=['house-fill','person-circle'],
                menu_icon='chat-text-fill',
                default_index=0,
                styles={
                    "container": {"padding": "0!important","background-color":'#660000'},
                    "icon": {"color": "blue", "font-size": "20px"}, 
                    "menu_title":{"background-color": "blue"} ,    
                    "nav-link": {"color":"white","font-size": "20px", "text-align": "center", "margin":"0px", "--hover-color": "#660000"},
                    "nav-link-selected": {"background-color": "green"},}
            )
        
    if app =="About Us":
            st.sucess("About Science For Social Good CIC")
            st.info ("#### Vocational training transforms lives and drives economic growth in developing countries.\
                       It equips individuals with the practical skills and knowledge needed to secure employment, start businesses, and contribute to their community’s development.\
                       Through our work, we constantly explore the significance of vocational training in developing countries and its positive impact on individuals, communities, and overall socio-economic progress.\
                       Our direction is genetic analysis testing verticals in Agriculture, Human Identification (Forensic DNA), quality and safety testing (environmental, food and water).\
                       To achieve this goal, SSG-CIC aims to establish Vocational Training Hubs (VTHs) across the globe and build local capacity for training and development.\
                       All of this is possible with your support and volunteering.  How you can support us?\
                       Our fundraising is through 4 main routes:\
                            1. Non Profit Consulting\
                            2. Crowdfunding\
                            3. Volunteer and Work with us\
                            4. Donate used laboratory equipment, IT Gear and Software")

            st.link_button("Click here to view more information", "https://www.rippleandco.com/about-us", type = "primary")
        
    if app == "AI Sample Assesement Tool":
            st.title("AI Assesement Tool for Forensic Sample Analysis")
            st.subheader("Assess Sample")
            st.write("### The SFSG AI Tool is a comprehensive and automated solution for forensic sample quality assessment.\
                        By combining structured decision-making with advanced machine learning techniques, it:\
                        •	Saves time and improves efficiency in forensic workflows.\
                        •	Provides accurate, reliable assessments.\
                        •	Enhances decision-making with clear, actionable insights.\
                            This tool is an invaluable asset for forensic labs, ensuring quality control and minimizing errors in sample analysis")
            sample_id = st.text_input("Enter Sample ID for Analysis")
            if sample_id:
                assess_sample(prepared_data, sample_id)
           

if __name__ == "__main__":
    main()

#def main():
    #st.title("AI Assesement Tool for Forensic Sample Analysis")

    #uploaded_file = st.file_uploader("Upload CSV File", type="csv")
    #if uploaded_file is not None:
        #df = pd.read_csv("SFSG_Dataset.csv")
        #st.write("### Uploaded Dataset")
        #st.write(df.head())

        #st.subheader("Data Preparation")
        #prepared_data = prepare_data(df)
        #st.write("### Processed Dataset")
        #st.write(prepared_data.head())

        #st.subheader("Assess Sample")
        #sample_id = st.text_input("Enter Sample ID for Analysis")
        #if sample_id:
            #assess_sample(prepared_data, sample_id)

        #st.subheader("Train Machine Learning Model")
        #if st.button("Train Model"):
            #train_model(prepared_data)






