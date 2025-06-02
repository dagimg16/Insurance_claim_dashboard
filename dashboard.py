import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
from db_utils import get_claim_by_id, update_liability_in_db
from model_utils import load_model, predict_fraud, preprocess_input, load_shap_explainer, st_shap

st.set_page_config(
    page_title="🔍 Insurance Claim Processor",
    layout="wide",
    initial_sidebar_state="expanded")

col = st.columns((10, 5), gap='medium')  

with col[0]:

    st.title("🔍 Insurance Claim Processor")
    st.write("Search a claim by ID to view its details and fraud prediction.")

    # User input
    claim_id = st.text_input("Enter Claim ID")
    st.markdown("---")

    if claim_id:
        claim = get_claim_by_id(claim_id)

        if claim is not None:
            claim_df = claim
            claim_df['policy_start'] = pd.to_datetime(claim_df['policy_start'])
            claim_df['incident_date'] = pd.to_datetime(claim_df['incident_date'])

            st.markdown("### 📄 Claim Information")
        
            col1, col2, col3 = st.columns(3)
            col1.metric("Policy Start", claim_df['policy_start'].dt.strftime('%Y-%m-%d').values[0])
            col2.metric("Incident Date", claim_df['incident_date'].dt.strftime('%Y-%m-%d').values[0])
            col3.metric("State", claim_df['state'].values[0])

            col4, col5, col6 = st.columns(3)
            col4.metric("Vehicle Type", claim_df['vehicle_type'].values[0])
            col5.metric("Reported By", claim_df['reported_by'].values[0])
            col6.metric("Policy Limit", f"${claim_df['policy_limit'].values[0]:,.2f}")

            col7, col8, col9 = st.columns(3)
            col7.metric("Insured Age", claim_df['insured_age'].values[0])
            col8.metric("Prior Claims", claim_df['prior_claims_count'].values[0])
            col9.metric("Claim Amount", f"${claim_df['claim_amount'].values[0]:,.2f}")

            st.divider()

            st.subheader("📝 Incident Details")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Incident Type**")
                st.info(claim_df['incident_type'].values[0])

            with col2:
                st.markdown("**Fact of Loss**")
                st.info(claim_df['fact_of_loss'].values[0])

            st.subheader("⚖️ Liability Adjustment")

            default_insured_liability = int(claim_df['insured_liability'].values[0]) 
            default_claimant_liability = int(claim_df['claimant_liability'].values[0]) 

            col1, col2 = st.columns(2)

            insured_liability = col1.number_input(
                "Insured Liability (%)", min_value=0, max_value=100, value=default_insured_liability, step=1)
            
            claimant_liability = col2.number_input(
                "Claimant Liability (%)", min_value=0, max_value=100, value=default_claimant_liability, step=1)

            if st.button("💾 Save Liability Decision"):
                if insured_liability + claimant_liability != 100:
                    st.error("The total liability must equal 100%!")
                else:
                    # Update in the database
                    update_liability_in_db(claim_id, insured_liability, claimant_liability)
                    st.success("Liability decision saved successfully!")

            with col[1]:
                st.subheader("🧠 Model Prediction")

                model = load_model()
                features = preprocess_input(claim)
                prediction, probability = predict_fraud(model, features)

                explainer = load_shap_explainer(model)
                shap_values = explainer.shap_values(features)
                feature_names = features.columns

                prediction = int(prediction)
                probability = float(probability)

                
                if prediction == 1:
                    st.error(f"⚠️ Fraudulent Claim (Confidence: {probability:.2f})")
                else:
                    st.success(f"✅ Legitimate Claim (Confidence: {1 - probability:.2f})") 

                st.subheader("🔍 Feature Importance (SHAP Explanation)")

                fig, ax = plt.subplots()
                shap.summary_plot(shap_values, features=features, feature_names=feature_names, plot_type="bar", max_display = 5, show=False)
                st.pyplot(fig)

                st.subheader("🧩 Detailed SHAP Force Plot")

            #     shap_html = shap.force_plot(
            #     explainer.expected_value,
            #     shap_values[0],
            #     features.iloc[0],
            #     feature_names=features.columns,
            #     matplotlib=False
            # )
                
                # visualize the training set predictions
                st_shap(shap.force_plot(explainer.expected_value, shap_values, features), 400)

        else:
            st.warning("❗ No claim found with that ID.")
