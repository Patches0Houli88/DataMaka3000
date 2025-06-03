import streamlit as st
import pandas as pd
import numpy as np
import random
from faker import Faker
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io

fake = Faker()

st.set_page_config(page_title="CSV Data Generator", layout="wide")
st.title("🧪 Custom CSV Data Generator")

st.sidebar.header("🛠️ Schema Builder")

column_names = []
column_types = []
data_types = [
    "name", "email", "int", "float", "float_normal", "bool", "date", "city", "state", "zip", "category", "company", "job", "phone", "gender"
]

category_label_input = st.sidebar.text_area(
    "Optional: Category Labels and Weights (format: label,weight per line)",
    value="A,0.4\nB,0.25\nC,0.2\nD,0.1\nE,0.05"
)

custom_categories = []
try:
    for line in category_label_input.strip().split("\n"):
        label, weight = line.split(",")
        custom_categories.append((label.strip(), float(weight)))
except:
    custom_categories = [('A', 0.4), ('B', 0.25), ('C', 0.2), ('D', 0.1), ('E', 0.05)]

with st.sidebar.form("schema_form"):
    num_rows = st.number_input("Number of Rows", min_value=10, max_value=10000, value=100)
    num_cols = st.number_input("Number of Columns", min_value=1, max_value=20, value=5)

    for i in range(int(num_cols)):
        col1, col2 = st.columns([2, 2])
        with col1:
            name = st.text_input(f"Column {i+1} Name", key=f"name_{i}")
        with col2:
            dtype = st.selectbox(f"Column {i+1} Type", data_types, key=f"dtype_{i}")

        column_names.append(name)
        column_types.append(dtype)

    submitted = st.form_submit_button("Generate Data")

# --- Enhanced Data Generator with Conditional Logic --- #
def generate_row(column_types):
    row = {}
    state_abbr = None
    for name, dtype in zip(column_names, column_types):
        if dtype == "gender":
            row[name] = random.choices(["male", "female"], weights=[0.5, 0.5])[0]
        elif dtype == "name":
            gender = row.get("gender", "male")
            row[name] = fake.name_male() if gender == "male" else fake.name_female()
        elif dtype == "email":
            row[name] = fake.email()
        elif dtype == "phone":
            row[name] = fake.phone_number()
        elif dtype == "company":
            row[name] = fake.company()
        elif dtype == "job":
            age = row.get("age") or row.get("Age") or 30
            row[name] = fake.job() if int(age) > 18 else None
        elif dtype == "int":
            row[name] = random.randint(18, 80)
        elif dtype == "float":
            row[name] = round(random.uniform(1000, 100000), 2)
        elif dtype == "float_normal":
            age = row.get("age") or row.get("Age") or 40
            mean = int(age) * 1200
            row[name] = max(0, round(np.random.normal(mean, 5000), 2))
        elif dtype == "bool":
            row[name] = random.choices([True, False], weights=[0.3, 0.7])[0]
        elif dtype == "date":
            row[name] = fake.date_between(start_date='-5y', end_date='today')
        elif dtype == "state":
            state_abbr = fake.state_abbr()
            row[name] = fake.state()
        elif dtype == "city":
            row[name] = fake.city()
        elif dtype == "zip":
            if not state_abbr:
                state_abbr = fake.state_abbr()
            row[name] = fake.postcode_in_state(state_abbr=state_abbr)
        elif dtype == "category":
            labels, weights = zip(*custom_categories)
            row[name] = random.choices(labels, weights=weights)[0]
        else:
            row[name] = None
    return row

# --- Main Execution --- #
if submitted:
    data = [generate_row(column_types) for _ in range(num_rows)]
    df = pd.DataFrame(data)

    st.success("✅ Data generated successfully!")

    # CSV Download
    st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode(), "generated_data.csv", "text/csv")

    # Excel Download
    output_excel = io.BytesIO()
    with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Data')
    st.download_button("⬇️ Download Excel", output_excel.getvalue(), "generated_data.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # JSON Download
    st.download_button("⬇️ Download JSON", df.to_json(orient="records", indent=2).encode(), "generated_data.json", "application/json")

    # --- Data Preview --- #
    st.subheader("📋 Data Preview")
    st.dataframe(df.head())

    # --- Auto Visual Summary --- #
    st.subheader("📈 Auto Visual Summary")
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            st.write(f"**{col}**")
            fig, ax = plt.subplots()
            df[col].hist(bins=20, ax=ax)
            st.pyplot(fig)
        elif df[col].dtype == 'object' or df[col].dtype == 'bool':
            st.write(f"**{col}**")
            st.bar_chart(df[col].value_counts())

    # --- Data Quality Report --- #
    st.subheader("🧪 Data Quality Report")
    st.write("**Null Values per Column:**")
    st.write(df.isnull().sum())
    st.write("**Duplicate Rows:**")
    st.write(df.duplicated().sum())
