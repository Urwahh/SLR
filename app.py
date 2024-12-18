import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Streamlit App Title
st.title("Dynamic Linear Regression App")


uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
   
    data = pd.read_csv(uploaded_file)
    
    st.write("### Uploaded Data Preview:")
    st.dataframe(data)
    st.write("### Select Columns for Regression")
    columns = data.columns.tolist()
    x_column = st.selectbox("Choose Feature (X):", columns)
    y_column = st.selectbox("Choose Label (Y):", columns)
    
    if x_column and y_column:
        # Prepare data for model
        X = data[[x_column]]
        Y = data[y_column]
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, Y_train)
        
        Y_pred = model.predict(X_test)
    
        st.write("### Model Results")
        st.write(f"Coefficient: {model.coef_[0]:.4f}")
        st.write(f"Intercept: {model.intercept_:.4f}")
        mse = mean_squared_error(Y_test, Y_pred)
        st.write(f"Mean Squared Error: {mse:.4f}")
        
        st.write("### Actual vs Predicted Visualization")
        fig, ax = plt.subplots()

        # Use X_test and Y_test for actual points
        ax.scatter(X_test, Y_test, color="blue", label="Actual")

        # Plot the predictions based on X_test
        ax.plot(X_test, Y_pred, color="red", label="Predicted")

        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        ax.legend()
        
        st.pyplot(fig)
