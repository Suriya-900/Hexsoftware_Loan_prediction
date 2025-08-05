import streamlit as st
import joblib
import numpy as np
import base64

# --- Load model and scaler ---
model = joblib.load('random_forest_model.pkl')

# --- Set background image using base64 ---
def set_background(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpeg;base64,{encoded}");
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;
                background-position: center;
            }}

            .main-title {{
                color: #FFD700;  /* Gold */
                text-shadow: 2px 2px 5px #b8860b;  /* Shadow to make it pop */
                font-size: 38px;
                font-weight: bold;
            }}

            .sub-title {{
                color: #ffdb58;  /* Lighter gold */
                font-size: 20px;
                font-weight: 500;
                text-shadow: 1px 1px 3px #000;
                font-style: italic;
            }}

            h3, h4, h5, h6 {{
                color: #ffcc70 !important;
                text-shadow: 1px 1px 3px #000;
            }}

            p, li, .markdown-text-container {{
                color: #e0f7fa !important;
                text-shadow: 1px 1px 2px #000;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )



# ✅ Set your local image path here
set_background("top-view-desk-with-financial-instruments.jpg")


# --- Page Title ---
st.markdown("<h1 class='main-title'>🎯 Loan Eligibility Prediction App</h1>", unsafe_allow_html=True)

# --- Page Navigation ---
page = st.sidebar.selectbox("Navigate", ["🏠 Home", "📊 Prediction"])

# --- Home Page ---
if page == "🏠 Home":
    st.markdown("<p class='sub-title'>This project predicts whether a loan applicant is likely eligible based on their financial profile.</p>", unsafe_allow_html=True)
    
    st.markdown("### 🔍 About This App")
    st.write("""
    This application uses a **Logistic Regression** model trained on loan applicant data to predict whether a user is likely to be eligible for a loan.

    ### ⚙️ Technologies Used:
    - Python
    - Streamlit
    - Scikit-learn
    - Pandas & NumPy
    - Joblib
    """)

    st.markdown("### 👨‍💻 Developed By:")
    st.markdown("**Suriya Vignesh**")

# --- Prediction Page ---
elif page == "📊 Prediction":
    st.markdown("### Please enter applicant details below:")

    # Input Fields
    gender = st.selectbox("Gender", ["Select Gender", "Male", "Female"])
    married = st.selectbox("Married", ["Select Marital Status", "Yes", "No"])
    dependents = st.selectbox("Dependents", ["Select Number of Dependents", "0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Select Education Level", "Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Select Employment Type", "Yes", "No"])
    applicant_income = st.number_input("Applicant Income", min_value=0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
    loan_term = st.number_input("Loan Term (in days)", min_value=0)
    credit_history = st.selectbox("Credit History", ["Select Credit History", "1.0", "0.0"])
    property_area = st.selectbox("Property Area", ["Select Property Area", "Urban", "Rural", "Semiurban"])

    # Prediction Button
    if st.button("🔮 Predict"):
        if "Select" in (gender, married, dependents, education, self_employed, credit_history, property_area):
            st.error("⚠️ Please fill in all fields correctly.")
        else:
            # Encoding categorical values
            gender = 1 if gender == "Male" else 0
            married = 1 if married == "Yes" else 0
            education = 1 if education == "Graduate" else 0
            self_employed = 1 if self_employed == "Yes" else 0
            credit_history = float(credit_history)
            dependents = 3 if dependents == "3+" else int(dependents)
            property_area = 2 if property_area == "Urban" else 1 if property_area == "Semiurban" else 0

            # Final input for prediction
            input_data = np.array([[gender, married, dependents, education, self_employed,
                                    applicant_income, coapplicant_income, loan_amount,
                                    loan_term, credit_history, property_area]])

            # Prediction (no scaling!)
            prediction = model.predict(input_data)

            # Output Result
            if prediction[0] == 1:
                st.success("✅ The applicant is likely **eligible** for the loan.")
            else:
                st.error("❌ The applicant is likely **not eligible** for the loan.")
