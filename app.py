import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# Streamlit App Title
st.title("Dynamic Linear Regression App")
# Step 1: File Upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    # Step 2: Load the uploaded CSV
    data = pd.read_csv(uploaded_file)
    
    st.write("### Uploaded Data Preview:")
    st.dataframe(data)
    # Step 3: Select Features (X) and Label (Y)
    st.write("### Select Columns for Regression")
    columns = data.columns.tolist()
    x_column = st.selectbox("Choose Feature (X):", columns)
    y_column = st.selectbox("Choose Label (Y):", columns)
    if x_column and y_column:
        # Prepare data for model
        X = data[[x_column]]
        Y = data[y_column]
        # Step 4: Train-Test Split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        # Step 5: Train Linear Regression Model
        model = LinearRegression()
        model.fit(X_train, Y_train)
        # Step 6: Make Predictions
        Y_pred = model.predict(X_test)
        # Step 7: Display Metrics
        st.write("### Model Results")
        st.write(f"Coefficient: {model.coef_[0]:.4f}")
        st.write(f"Intercept: {model.intercept_:.4f}")
        mse = mean_squared_error(Y_test, Y_pred)
        st.write(f"Mean Squared Error: {mse:.4f}")
        # Step 8: Visualization
        st.write("### Actual vs Predicted Visualization")
        fig, ax = plt.subplots()
        ax.scatter(X, Y, color="blue", label="Actual")
        ax.plot(X, model.predict(X), color="red", label="Predicted")
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        ax.legend()
        st.pyplot(fig)
