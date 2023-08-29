import streamlit as st

st.set_page_config(
    page_title="Cost of Care Service",
    page_icon='/home/oem/machine_learning/pages/healthcare.png'
)

st.write("Welcome to the beta app 👋")
st.sidebar.success("Select a demo above.")

st.markdown("""
            A linear regression model was created to forecast the cost of both social workers 
            and community connections as part of the investigation. This will help to predict 
            the value of preventive measures and how quantitative monetary estimation can be 
            used to make more informed decisions..
            **👈 Select a model type and care service prediction from the sidebar**
            Though their is still need for improving the model because of the size of the dataset.
    """)