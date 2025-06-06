import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import text
import shap
from db_utils import get_claim_by_id, update_liability_in_db, get_matching_claim_id
from model_utils import load_model, predict_fraud, preprocess_input, load_shap_explainer, st_shap
from adjuster_ai_assistant import qa_chain

st.set_page_config(
    page_title="🔍 Insurance Claim Processor",
    layout="wide",
    initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    /* Style for each column */
    div[data-testid="column"] > div {
        background-color: #f9f9f9; /* light background */
        padding: 2rem 1rem;
        border: 1px solid #d3d3d3;
        border-radius: 12px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.05);
        height: 100%; /* full height box */
    }

    /* Optional: Style page to center everything better */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

col = st.columns((5, 10, 5), gap='medium')  

with col[1]:
    with st.container():
        st.title("🔍 Insurance Claim Processor")
        st.write("Search a claim by ID to view its details and fraud prediction.")

        # User input
        # claim_id = st.text_input("Enter Claim ID")
        typed_input = st.text_input("Start typing a Claim ID:", key="claim_input_db")

        claim_id = None

        if typed_input and len(typed_input) >= 2:
            # SQL Query to fetch top 10 similar claim_ids
            query = text("""
                SELECT claim_id
                FROM claims
                WHERE claim_id ILIKE :typed_input
                ORDER BY claim_id
                LIMIT 10;
            """)

            # Execute query — wrap typed input with wildcards %
            params = {"typed_input": f"%{typed_input}%"}

            matching_claims = get_matching_claim_id(query, params)

            if not matching_claims.empty:
                options = ["🔍 Please select a Claim ID"] + matching_claims['claim_id'].tolist()
                selected_claim_id = st.selectbox("Select a Claim ID:", options)

                if selected_claim_id != "🔍 Please select a Claim ID":
                    claim_id = selected_claim_id
            else:
                st.info("No matches found. Try typing more.")

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

                with col[2]:
                    with st.container():
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
                        st.markdown("---")
                        st.subheader("🧩 Detailed SHAP Force Plot")

                        st_shap(shap.force_plot(explainer.expected_value, shap_values, features), 130)
                
            else:
                st.warning("❗ No claim found with that ID.")
with col[0]:
    with st.container():
        st.subheader("🚗 Auto Insurance Adjuster Assistant")
        st.markdown("""
                        Ask me anything about:
                        - Claims Handling Procedures 📝
                        - Liability Rules ⚖️
                        - Total Loss (TL) Thresholds 🚗
                        - State-Specific Guidelines 🌎
                        """)
        
        if "messages" not in st.session_state:
            st.session_state["messages"] = []  # List to hold all the messages

        # Chat Input
        user_input = st.text_input("Type your question here...", key="chat_input")

        if st.button("Send"):
            if user_input:
                # Append user message to the history
                st.session_state["messages"].append({"role": "user", "content": user_input})
                
                with st.spinner("Thinking..."):
                    # Get response from the qa_chain
                    ai_response = qa_chain.run(user_input)
                
                # Append AI response to the history
                st.session_state["messages"].append({"role": "assistant", "content": ai_response})

        if st.button("Clear Chat"):
            st.session_state["messages"] = []        
        # Display the full chat history
        for msg in st.session_state["messages"]:
            if msg["role"] == "user":
                st.markdown(f"🧑‍💻 **You:** {msg['content']}")
            else:
                st.markdown(f"🤖 **Assistant:** {msg['content']}")        