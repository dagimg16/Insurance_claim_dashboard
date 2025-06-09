# 🚗 AI-Powered Insurance Claim Handling Dashboard

## End-to-End Machine Learning and AI Solution for Faster, Smarter Claims Management

---

## 📚 Project Overview

Insurance claim handling is a complex and time-sensitive process that requires careful fraud detection and adherence to detailed state and company policies. Traditionally, it is slow, heavily manual, and prone to human error.

This project demonstrates how Machine Learning (ML) and AI Assistants can be integrated to enhance claim handling efficiency, improve fraud detection accuracy, and provide instant knowledge support for claim adjusters.

## 🎯 Project Goals

* **Automate Fraud Detection**: Predict the likelihood of a claim being fraudulent using machine learning.
* **Assist Adjusters with AI**: Provide instant, accurate answers to claim handling questions using an AI Assistant trained on internal policy documents.
* **Streamline Claim Processing**: Build a user-friendly dashboard to make fraud predictions and AI assistance accessible to claim handlers.

## 🛠️ Key Features

### 1. Synthetic Dataset Generation

* **Challenge**: Real claim data is sensitive and unavailable.
* **Solution**: Generated a realistic insurance claim dataset using:

  * **Faker**: To create initial claim profiles.
  * **SDV (Synthetic Data Vault)**: To statistically model and expand the dataset.

**Dataset Includes**:

* Claim ID, policy dates, incident details
* State, vehicle type, incident type, fact of loss
* Claim amount, policy limit, reported by (insured or claimant)
* Insured age, prior claims count
* Fraud flag (labeled using business logic)

### 2. Fraud Labeling Logic

Real-world inspired business rules were used to label claims as fraudulent:

* High claim amount relative to policy limit (>80%).
* Multiple prior claims (>2).
* Short time between policy start and incident (<30 days).
* Reported by claimant (instead of insured).
* Specific incident types (e.g. Theft, Weather Damage).

Fraud Score was computed and claims were labeled as fraud (1) if exceeding a threshold.

### 3. Machine Learning Model Training

* **Objective**: Predict fraud accurately.
* **Models Trained**:

  * Logistic Regression
  * Decision Tree
  * Random Forest
  * Support Vector Machine (SVM)
  * XGBoost — **best performer**

**Evaluation Metrics**:

![Accuracy and F1 Score](https://github.com/dagimg16/Insurance_claim_dashboard/blob/development/screenshots/accuracy_and_f1_score.png)

Accuracy and F1 Score (main focus due to fraud class imbalance)

![Precision and Recall](https://github.com/dagimg16/Insurance_claim_dashboard/blob/development/screenshots/recall_and_precision.png)

Precision and Recall


**Why XGBoost**:

* Best F1 Score: 0.83
* Balanced fraud detection with minimal false positives.
* Robust for imbalanced datasets.

### 4. Model Deployment

* Final model saved as a `.pkl` file.
* Integrated into a Streamlit dashboard.
* Real-time fraud prediction:

  * Search a claim by ID.
  * Load claim details from PostgreSQL database.
  * Preprocess features.
  * Predict fraud probability using the trained model.
  * Show result: Fraud or Not Fraud.

### 5. Explainable with SHAP

* **SHAP (SHapley Additive exPlanations)**:

  * Feature importance summary plots.
  * SHAP force plots for individual claim predictions.

Helps adjusters understand why a claim was flagged — improves trust and transparency.

### 6. AI Assistant for Adjusters

* **Problem**: Adjusters need fast, reliable access to claim handling rules.
* **Solution**: Built an AI Assistant that:

  * Ingests an internal Claims Handling Manual (sample PDF).
  * Splits the document into text chunks.
  * Creates embeddings using OpenAI Embeddings.
  * Stores vectors in a FAISS Vector Database.
  * Retrieves relevant chunks based on the adjuster's question.
  * Uses GPT-3.5 to summarize and answer based on the manual.

PDF ➔ Text Split ➔ Embeddings ➔ FAISS Vector Store ➔ Question ➔ Similar Chunks ➔ GPT-3.5 ➔ Answer

**Technology Stack**:

* LangChain for document processing.
* OpenAI Embeddings and GPT-3.5.
* FAISS for fast vector search.

### 7. Streamlit Dashboard

**Features**:

* Search for claims dynamically with dropdown auto-suggestions.
* View detailed claim information.
* Get real-time fraud predictions.
* Visualize SHAP-based model explanations.
* Use the built-in AI Assistant for policy questions.

## 📦 Project Structure

```
├── adjuster_ai_assistant.py     # AI Assistant (RAG + LLM)
├── dashboard.py                 # Main Streamlit app
├── db_utils.py                  # Database helper functions
├── model_utils.py               # Model loading and prediction functions
├── load_data_to_PostgreSQL.py   # Script to load synthetic data to PostgreSQL
├── model_train_and_test.ipynb   # ML training notebook
├── synthetic_data_generation.ipynb # Data generation notebook
├── fraud_model_config.json      # Model configuration file
├── claims_dataset.csv           # Synthetic claims data (optional)
├── model/                       # Folder containing trained model (.pkl)
├── adjuster_manual/             # Folder containing Adjuster Manual PDF
├── adjuster_manual_vectorstore/ # Stored document embeddings for AI Assistant
├── screenshots/                 # Folder containing demo screenshots
├── data/                        # data assets
├── mlruns/                      # MLFlow tracking folder
├── requirements.txt             # Project dependencies
├── .env.example                 # Environment variable sample file
├── .gitignore                   # Git ignore rules
└── README.md                    # Project documentation
```

## 🚀 Future Improvements

* Deploy dashboard on Streamlit Cloud or AWS.
* Add user authentication for secure access.
* Fine-tune the AI Assistant with more domain-specific documents.
* Integrate real-time claim data ingestion.

## 📚 Key Technologies Used

* Python (Faker, SDV, pandas, scikit-learn, XGBoost)
* Machine Learning (Classification, Model Evaluation)
* SHAP (Explainable AI)
* Streamlit (Dashboard)
* PostgreSQL (Database)
* LangChain (Document processing pipeline)
* OpenAI API (GPT-3.5 and Embeddings)
* FAISS (Vector Store)

## 📄 License

This project is for educational and demonstration purposes.

## 🔧 Usage Instructions

1. **Clone the Repository**

```
git clone <repository-url>
cd <repository-directory>
```

2. **Install Required Packages**

It's recommended to use a virtual environment.

```
pip install -r requirements.txt
```

3. **Set Up PostgreSQL Database**

* Make sure you have a local PostgreSQL server running.
* Create a new database (e.g., `insurance_claims_db`).

4. **Create Your `.env` File**

Inside the project root, create a `.env` file with the following variables (you can reference the provided `.env.example`):

```
DB_USER=your_db_username
DB_PASSWORD=your_db_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=insurance_claims_db
OPENAI_API_KEY=your_openai_api_key
```

**Note**: You will need an OpenAI API Key for the AI Assistant. You can get one from OpenAI.

5. **Load Data into PostgreSQL**

Run the following script to load the claims dataset into your database:

```
python load_data_to_PostgreSQL.py
```

6. **Run the Streamlit Dashboard**

```
streamlit run dashboard.py
```

## 📷 Screenshots
![Screenshot of home page](https://github.com/dagimg16/Insurance_claim_dashboard/blob/development/screenshots/homepage.png)
Search a claim by ID
![Screenshot of home page](https://github.com/dagimg16/Insurance_claim_dashboard/blob/development/screenshots/claim_screen.png)
See the claim profile loaded, get fraud prediction and SHAP explanation and ask a policy-related question to the AI Assistant and see an instant answer.
![Screenshot of home page](https://github.com/dagimg16/Insurance_claim_dashboard/blob/development/screenshots/claim_details.png)



## 📄 License

This project is for educational and demonstration purposes.
