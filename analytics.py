import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os
st.set_option('deprecation.showPyplotGlobalUse', False)

script_path = os.path.abspath(os.path.dirname(__file__))
file_path = os.path.join(script_path, 'correct_data (1).xlsx')

@st.cache
def load_data(path):
    df = pd.read_excel(path)
    df = df.drop(columns=['Title', 'Surname', 'Contact Telephone Number', 'Referred From', 'Date Referral Received', 'Referred To', 'Date Referral Received',
                      'Referred To', 'Alternative Contact Telephone Number', 'Address 1', 'Postcode', 'Referred To', 'Referred From',
                      'Address 2', 'National Insurance Number', 'Email Address', 'End Leaving Questionnaire Completed.', 'End Leaving Questionnaire Completed.',
                      'Support Officer', 'First Name(s)', 'Date Referred', 'Understand Welsh', 'Speak Welsh', 'Read Welsh', 'Write Welsh'], axis=1)
    df = df.fillna("")

    df['Dob'] = pd.to_datetime(df['Date of Birth'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df['Year'] = df['Dob'].dt.year
    df['Working_Days'] = (df['Completion Date'] - df['Project Start Date']).apply(lambda x: x.days + 1)

    df = df.drop(columns=['Date of Birth', 'Dob', 'Project Start Date', 'Completion Date'], axis =1)
    pd.to_numeric(df['Year'])
    df['Age'] = 2023 - df['Year']

    df = df.drop(columns=['Year'])
    df['charges per hour £'] = 13
    df['Average Time Spent'] = 5
    df['Community_Connector_charges'] = 16.83
    df['Social_Worker_charges'] = 18.55

    df.to_csv("processed_data.csv")
    return df

data_load_state = st.text('Loading data...')
df1 = load_data(file_path)
data_load_state.text("Done! (using st.cache)")


st.title("Data Analysis and Visualization")

data_load_state = st.text('Loading data...')
df1 = pd.read_csv('processed_data.csv')
df1 = df1.drop(columns=['Preferred Language for Communication', 'Cluster Area'], axis =1)
data_load_state.text("Done! (using st.cache)")

ata_load_state = st.text('Statistical Summary Analysis...')
df1.describe().T
data_load_state.text("Done! (using st.cache)")

st.subheader("Processed Data")
# st.write(df1)

df1 = df1.fillna(0)
df1 = df1.assign(cost_of_care=pd.Series(df1['charges per hour £'] * df1['Working_Days'] * df1['Average Time Spent']))
df1 = df1.assign(cost_of_commConnector=pd.Series(df1['Community_Connector_charges'] * df1['Working_Days'] * df1['Average Time Spent']))
df1 = df1.assign(cost_of_socialWorker=pd.Series(df1['Social_Worker_charges'] * df1['Working_Days'] * df1['Average Time Spent']))

st.subheader("Average Cost of Care by Intervention")
df1_group = df1.groupby(['Town/City', 'Gender'])['cost_of_care'].mean().reset_index()
df1_filtered_ = df1_group[df1_group != 0]
fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.barplot(data=df1_filtered_, x='Town/City', y='cost_of_care', hue='Gender')
plt.title('Average Cost of Care by Town/City and Gender')
plt.xlabel('Town/City')
plt.ylabel('Average Cost of Care £')
plt.xticks(rotation=45)
st.pyplot(fig1)

# Visualization 2: Average Cost of Care by Professionals Visited
st.subheader("Average Cost of Care by Professionals Visited")
df1_group = df1.groupby(['Individual Visited/Joint Visited with Professionals'])['cost_of_care'].mean().reset_index()
df2_filtered_ = df1_group[df1_group != 0]
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.barplot(data=df2_filtered_, x='Individual Visited/Joint Visited with Professionals', y='cost_of_care')
plt.title('Cost')
plt.xlabel('Individual Visited/Joint Visited with Professionals')
plt.ylabel('Average Cost of Care £')
plt.xticks(rotation=45)
st.pyplot(fig2)  # Display the Seaborn plot in Streamlit

st.subheader('Cost Analysis for Cost of Care by Town/City (Non-Zero')
df1_grouped = df1.groupby('Town/City')['cost_of_care'].agg(['sum', 'mean', 'min']).reset_index()
df1_grouped_filtered = df1_grouped[df1_grouped['Town/City'] != 0]
# Set custom palette colors
custom_palette = sns.color_palette(['#1f77b4', '#ff7f0e', '#2ca02c'])
# Plotting using Seaborn
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.barplot(data=df1_grouped_filtered, x='Town/City', y='sum', color=custom_palette[0], label='Total Cost')
sns.barplot(data=df1_grouped_filtered, x='Town/City', y='mean', color=custom_palette[1], label='Average Cost')
sns.barplot(data=df1_grouped_filtered, x='Town/City', y='min', color=custom_palette[2], label='Minimum Cost')
plt.title('Cost Analysis for Cost of Care by Town/City (Non-Zero)')
plt.xlabel('Town/City')
plt.ylabel('Cost £')
plt.xticks(rotation=45)
plt.legend()
st.pyplot(fig3)