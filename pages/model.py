import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import os
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)

script_path = os.path.abspath(os.path.dirname(__file__))
file_path = os.path.join(script_path, 'com_conn_data.csv', 'social_worker_data.csv')

def load_data(dataset_path, target_col_name):
    data = pd.read_csv(dataset_path)
    columns_to_drop = ['Unnamed: 0', 'cost_of_care', target_col_name]
    
    if target_col_name == 'cost_of_commConnector':
        columns_to_drop.append('cost_of_commConnector')
    else:
        columns_to_drop.append('cost_of_socialWorker')
    
    X = data.drop(columns=columns_to_drop)
    y = data[target_col_name]
    
    return X, y

def train_model(X_train, y_train, model_type, n_estimators):
    if model_type == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=n_estimators)
    
    model.fit(X_train, y_train)
    return model

def main():
    st.title("Care Service Cost Prediction")
    html_temp = """
    <div style="background-color: tomato;padding:10px">
    <p style='color:white;text-align:center;'>Check the cost of care service by using either a Community Care Service or using a Social Worker Service..</p>
    </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)

    datasets_name = st.sidebar.selectbox("Select Dataset for Prediction", ("Com Conn", "Social Worker"))
    target_col_name = "cost_of_commConnector" if datasets_name == "Com Conn" else "cost_of_socialWorker"

    model_type = st.sidebar.select_slider("Select Model", ("Linear Regression", "Random Forest"))
    n_estimators = st.sidebar.slider("Number of Estimators", 1, 100, 10)
    test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.3, 0.05)

    X, y = load_data(f"pages/{datasets_name.lower().replace(' ', '_')}_data.csv", target_col_name)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    model = train_model(X_train, y_train, model_type, n_estimators)

    work_days = st.number_input("Work Days", value=0, step=1)
    Age = st.number_input("Age", value=0, step=1)
    charge_per_hour = st.number_input("Charges per Hour", value=0.0, step=0.01)
    Average_Time_Spent = st.number_input("Average Time Spent", value=0.0, step=0.01)
    cost_of_commConnector = st.number_input("Cost of Community Connector", value=0.0, step=0.01)

    try:
        work_days = int(work_days)
        Age = int(Age)
        charge_per_hour = float(charge_per_hour)
        Average_Time_Spent = float(Average_Time_Spent)
        cost_of_commConnector = float(cost_of_commConnector)

        input_data = [[work_days, Age, charge_per_hour, Average_Time_Spent, cost_of_commConnector]]
        result = model.predict(input_data)
        
        st.subheader("Prediction Result")
        st.write(f"The estimated cost of care is £ {result[0]:.2f}")
        
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred) * 100
        
        st.subheader("Model Evaluation")
        st.write(f"The confidence level of the model is: {score:.2f}%")
        
        plt.scatter(y_test, y_pred)
        plt.xlabel("Actual result")
        plt.ylabel("Predicted result")
        plt.title("Actual vs Predicted")
        st.pyplot()
    except ValueError:
        st.error("Please enter valid numeric values.")

if __name__ == "__main__":
    main()
