import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import streamlit as st

# ---------------- LOAD DATA ----------------
df = pd.read_excel("university.xlsx")

X = df.drop(["Univ", "State"], axis=1)

# ---------------- SCALE DATA ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- TRAIN MODEL ----------------
model = KMeans(n_clusters=3, random_state=42)
model.fit(X_scaled)

# ---------------- STREAMLIT UI ----------------
st.title("University Clustering App")

st.write("Enter university details to find its cluster")

sat = st.number_input("SAT Score", 800, 1600, 1200)
top10 = st.number_input("Top 10 Percentage", 0, 100, 50)
accept = st.number_input("Acceptance Rate", 0, 100, 50)
sfr = st.number_input("Student Faculty Ratio", 1, 30, 15)
expenses = st.number_input("Annual Expenses", 5000, 70000, 20000)
grad = st.number_input("Graduation Rate", 0, 100, 80)

if st.button("Predict Cluster"):
    user_data = np.array([[sat, top10, accept, sfr, expenses, grad]])
    user_scaled = scaler.transform(user_data)
    cluster = model.predict(user_scaled)

    st.success(f"This university belongs to Cluster {cluster[0]}")
