============================================================================== ⚡ FOUNDATION PROJECT (GROUP 10): UTILITY ENERGY SALES FORECASTING

Deployment & Monitoring Module | Built with CRISP-ML(Q)

This document serves as the Master Guide for the project. It covers the
architecture, file descriptions, step-by-step execution, and the demo script.

📂 1. FILE DESCRIPTIONS 

main_FP.py

The "Brain" (Backend). A FastAPI server that loads the model.
FEATURES: Implements Real-time Data Drift Detection (KS/PSI Proxy).
LOGIC: Checks if inputs are within historical 99% Confidence Intervals.

ui_FP.py

The "Face" (Frontend). A Streamlit dashboard for users.
FEATURES: "Smart Input" (auto-calculates calendar days), Scenario Planning.
ALERTS: Displays a Red Warning Box if the Backend detects drift.

regenerate_model_v2.py

The "Factory". A script to train and save the XGBoost model locally.
PURPOSE: Solves "Version Mismatch" errors by ensuring the model is built
using the exact same Python/XGBoost version as your Docker container.

.dockerignore

Critical optimisation. Tells Docker to ignore heavy files (venv, git, raw data) to make builds fast.

Dockerfile

The "Recipe". Tells Docker how to build the environment.
BASE: Uses Python 3.12-slim to match your local development setup.


master_df.csv

The "Memory". The Master datatset. Having almost 300 data points.
This historical training data used to calculate Drift Thresholds.

xgboost_model.pkl

The "Artifact". The trained model file (Output of regenerate_model_v2.py).

requirements.txt

The "Ingredients". Lists all Python libraries (fastapi, xgboost>=3.0.0).

docker-compose.yml

The "Orchestrator". Defines two microservices (frontend & backend).
CONFIG: Maps ports 8000 (API) and 8501 (UI) to your localhost.



🛠️ 2. STEP-BY-STEP EXECUTION PROCESS

Follow this sequence exactly to run the project from scratch.

[PHASE 1: MODEL GENERATION]
Goal: Create a model file compatible with your system.

Open your Jupiter terminal in the project folder.

Run the generation script:

python regenerate_model_v2.py

Wait ~30 seconds.

It will print "System Version Check".

It will perform Grid Search.

It will save 'xgboost_model.pkl'.

It will generate 3 plots (.png) in the folder.

[PHASE 2: DEPLOYMENT]
Goal: Launch the Microservices Architecture.

Ensure Docker Desktop is running (Green Whale Icon).

Run the build command:

docker-compose down
docker-compose up --build

Wait for the logs to stop scrolling and show colorful text.

 "Uvicorn running on http://0.0.0.0:8000"

 "Streamlit ... http://0.0.0.0:8501"

[PHASE 3: ACCESS]
7.  Open your web browser.
8.  Go to: http://localhost:8501




🎥 3. DEMO 



Scene 1: The Setup (Architecture)



/////We deployed the solution using a containerized microservices architecture.
The FastAPI backend handles inference and monitoring, while Streamlit provides
the scenario planning interface.

Scene 2: The Logic (Smart Inputs)

Show the "Forecast Period" section in the UI. Change the date.

/////To reduce user error, we automated feature engineering. The system
automatically extracts calendar features like Saturdays, Sundays, and Holidays
to capture grid load nuances.

Scene 3: The Normal Prediction

Enter: Price=10.5, Temp=75, Fuel=3000. Click Predict.

Result: Green Success Box.

////For standard operational conditions, the model predicts sales of ~26,000 mkWh.
The system status is Normal.

Scene 4: The "CRISP-ML(Q)" Feature (Drift Detection)

Enter: Temp=120 (Extreme Heat). Click Predict.

Result: Red Alert Box.

/////Crucially, we implemented Real-time Data Drift Detection. Here, the system
detects that 120°F is outside the training distribution (99% Confidence Interval).
It triggers a System Alert and logs the event for auditing, ensuring we don't
blindly trust the model during anomalies.

📊 4. TECHNICAL SPECS

Model Accuracy (MAPE): ~1.66%

Model Type: XGBoost Regressor (v3.1.2)

Input Features: 14 (Revenue removed to prevent Data Leakage)

Drift Detection: Statistical Thresholding (KS/PSI Proxy Logic)

Audit Trail: drift_logs.log (Generated automatically)

============================================================================== END OF GUIDE