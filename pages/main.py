import streamlit as st
import pickle

data_load_state = st.text('Loading model...')
pickle_in = open("/home/oem/machine_learning/pages/conn_cost-model.pk1", 'rb')
model = pickle.load(pickle_in)
data_load_state.text("Done! (using st.cache)")

@st.cache
def predictions(work_days, Age, charge_per_hour, Average_Time_Spent, cost_of_care, cost_of_commConnector):
    input_data = [[work_days, Age, charge_per_hour, Average_Time_Spent, cost_of_care, cost_of_commConnector]]
    predictions = model.predict(input_data)
    return predictions[0]

def main():
    st.title("Care Service Cost")
    html_temp = """
    <div style="background-color: tomato;padding:10px">
    <h2 style='color:white;text-align:center;'>Care Service Cost Predictions </h2>
    </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)
    work_days = st.text_input("work_days")
    Age = st.text_input("Age")
    charge_per_hour = st.text_input("Charge per hour")
    Average_Time_Spent = st.text_input("Average time")
    # cost_of_care = st.text_input("cost_of_care")
    Community_Connector_charges = st.text_input("Community_Connector_charges")

    try:
        work_days = float(work_days)
        Age = float(Age)
        charge_per_hour = float(charge_per_hour)
        Average_Time_Spent = float(Average_Time_Spent)
        cost_of_care = float(cost_of_care)
        cost_of_commConnector = float(Community_Connector_charges)

        result = ""
        if st.button("Predictions"):
            result = predictions(work_days, Age, charge_per_hour, Average_Time_Spent, cost_of_care, cost_of_commConnector)
        st.success("The output is: {:.2f}".format(result))
    except ValueError:
        st.error("Please enter valid numeric values.")

if __name__ == "__main__":
    main()