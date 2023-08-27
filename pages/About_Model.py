import streamlit as st

st.set_page_config(
    page_title="Cost of Care Service",
    page_icon='/home/oem/machine_learning/pages/healthcare.png'
)

st.write("Welcome to the beta app 👋")
st.sidebar.success("Select a demo above.")

st.markdown("""
A linear regression model was created to forecast the cost of both social workers and community connections as part 
            of the investigation. Linear Regression Measures the association between two or more variables. 
            A dependent variable is predicted using this modeling approach based on one or more independent factors..
            **👈 Select a model type and care service prediction from the sidebar** 
    """)