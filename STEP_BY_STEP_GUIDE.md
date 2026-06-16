# 🚀 STEP-BY-STEP GUIDE: Run Banking Transaction Prediction on Your Laptop

> **Last Updated:** June 2026 | **Estimated Time:** 30-45 minutes | **Difficulty:** Beginner-Friendly

---

## 📋 BEFORE YOU START - Checklist

Make sure you have:
- [ ] A laptop with Windows 10/11, macOS, or Linux
- [ ] At least **8GB RAM** (16GB recommended)
- [ ] **5GB free disk space**
- [ ] Internet connection (to download packages)
- [ ] Basic familiarity with Command Prompt / Terminal

---

## STEP 1: Install Python (5 minutes)

### For Windows Users:

**Option A: Microsoft Store (Easiest)**
1. Click **Start** → Search **"Microsoft Store"**
2. Search **"Python 3.12"** (or latest version)
3. Click **Get** → **Install**
4. Open **Command Prompt** (search "cmd")
5. Type: `python --version`
6. You should see: `Python 3.12.x` ✅

**Option B: Official Installer**
1. Go to: https://www.python.org/downloads/
2. Click **"Download Python 3.12.x"**
3. Run the downloaded `.exe` file
4. ⚠️ **IMPORTANT:** Check **"Add Python to PATH"** at the bottom
5. Click **"Install Now"**
6. Open Command Prompt, type: `python --version`

### For Mac Users:

**Option A: Official Installer (Recommended)**
1. Go to: https://www.python.org/downloads/
2. Click **"Download Python 3.12.x"** for macOS
3. Open the downloaded `.pkg` file
4. Follow installation wizard
5. Open **Terminal** (Cmd + Space, type "Terminal")
6. Type: `python3 --version`
7. You should see: `Python 3.12.x` ✅

**Option B: Homebrew**
```bash
# Open Terminal, run these commands:
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python
python3 --version
```

### For Linux (Ubuntu/Debian) Users:

```bash
# Open Terminal, run these commands:
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-pip
python3.12 --version
```

---

## STEP 2: Install pip (Python Package Manager)

### Windows:
```cmd
python -m ensurepip --upgrade
python -m pip --version
```

### Mac/Linux:
```bash
python3 -m ensurepip --upgrade
python3 -m pip --version
```

You should see something like: `pip 24.x` ✅

---

## STEP 3: Download the Project (2 minutes)

### Option A: Download ZIP (Easiest for beginners)

1. The project folder is at: `/mnt/agents/output/banking_transaction_prediction/`
2. Download it as a ZIP file from your file manager
3. Extract it to your **Desktop** or **Documents** folder
4. Rename the folder to: `banking-prediction` (shorter name, easier to type)

### Option B: Using Git (If you know Git)

```bash
git clone https://github.com/yourusername/banking-transaction-prediction.git
cd banking-transaction-prediction
```

---

## STEP 4: Open Terminal/Command Prompt in Project Folder

### Windows:
1. Open **File Explorer**
2. Navigate to your `banking-prediction` folder
3. Click in the address bar at the top
4. Type `cmd` and press **Enter**
5. Command Prompt opens with this folder as the current location

### Mac:
1. Open **Finder**
2. Navigate to your `banking-prediction` folder
3. Right-click → **"New Terminal at Folder"**
   (If not available: Go to System Preferences → Keyboard → Shortcuts → Services → Enable "New Terminal at Folder")

### Linux:
1. Open **File Manager**
2. Navigate to the folder
3. Right-click → **"Open in Terminal"**

---

## STEP 5: Create Virtual Environment (3 minutes)

A virtual environment keeps this project's packages separate from your other projects. This prevents conflicts.

### Windows:
```cmd
python -m venv venv
```

### Mac/Linux:
```bash
python3 -m venv venv
```

You should see a new `venv` folder created in your project directory. ✅

---

## STEP 6: Activate Virtual Environment (1 minute)

### Windows (Command Prompt):
```cmd
venv\Scripts\activate
```

### Windows (PowerShell):
```powershell
venv\Scripts\Activate.ps1
```

### Mac/Linux:
```bash
source venv/bin/activate
```

**You'll know it's working when you see `(venv)` at the beginning of your command line:**
```
(venv) C:\Users\YourName\banking-prediction>
```

---

## STEP 7: Install Required Packages (10-15 minutes)

Make sure you're in the project folder and `(venv)` is showing, then run:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install all 40+ packages including:
- numpy, pandas (data handling)
- scikit-learn, xgboost, lightgbm, catboost (machine learning)
- shap (model interpretability)
- flask, streamlit (web apps)
- matplotlib, seaborn, plotly (visualization)

**⚠️ This takes 10-15 minutes depending on your internet speed. Don't interrupt it.**

If you see any red errors, don't worry - some packages have optional dependencies. As long as the main ones install, you're fine.

To verify installation:
```bash
python -c "import pandas, numpy, sklearn, xgboost, lightgbm, flask, streamlit; print('All packages installed successfully!')"
```

---

## STEP 8: Run the Complete ML Pipeline (10 minutes)

This is the main step that trains all 12 machine learning models and generates reports.

```bash
python main_pipeline.py --phase all --data data/raw/train.csv
```

**What happens:**
1. ✅ Loads the dataset (200,000 customers, 200 features)
2. ✅ Validates data quality
3. ✅ Engineers new features (statistical aggregations, interactions)
4. ✅ Selects best features using 6 methods
5. ✅ Trains 12 ML models with cross-validation
6. ✅ Tunes hyperparameters with Optuna
7. ✅ Evaluates models (ROC curves, confusion matrices, lift charts)
8. ✅ Generates SHAP interpretability plots
9. ✅ Saves the best model

**Output you'll see:**
```
Training: LogisticRegression... CV AUC = 0.8234
Training: RandomForest... CV AUC = 0.8923
Training: XGBoost... CV AUC = 0.9189
Training: LightGBM... CV AUC = 0.9234
BEST MODEL: LightGBM (ROC-AUC: 0.9234)
```

---

## STEP 9: View the Results (2 minutes)

After the pipeline completes, check these folders:

### 1. View Generated Charts
```
reports/figures/
├── 01_missing_target_analysis.png
├── 02_feature_distributions.png
├── 03_outlier_skewness_analysis.png
├── 04_correlation_analysis.png
├── roc_curve_LightGBM.png
├── pr_curve_LightGBM.png
├── confusion_matrix_LightGBM.png
├── lift_gain_LightGBM.png
├── shap_summary_LightGBM.png
└── permutation_importance_LightGBM.png
```

Just double-click any `.png` file to view it.

### 2. Read Reports
```
reports/
├── 02_eda_report.md           ← EDA findings
└── technical_report.md         ← Full technical details
```

Open with any text editor or Markdown viewer.

### 3. Check Saved Models
```
models/
└── artifacts/
    └── [timestamp]/
        ├── model.pkl            ← Best trained model
        ├── feature_selector.pkl ← Feature selection rules
        └── scaler.pkl           ← Scaling parameters
```

---

## STEP 10: Launch the Interactive Dashboard (2 minutes)

```bash
streamlit run app/streamlit/app.py
```

**What happens:**
- Your browser opens automatically at `http://localhost:8501`
- You see a professional banking dashboard
- Navigate through tabs: Dashboard, Single Prediction, Batch Prediction, Model Performance, About

**To stop the dashboard:** Press `Ctrl + C` in the terminal.

---

## STEP 11: Start the Prediction API (2 minutes)

Open a **NEW** terminal window (keep the dashboard running in the first one):

```bash
# Navigate to project folder again
cd path/to/banking-prediction

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Start API
python app/flask/app.py
```

**Test the API:**

Open your browser and go to: `http://localhost:5000/health`

You should see:
```json
{"status": "healthy", "model_loaded": true}
```

**Test a prediction:**

Open another terminal and run:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"var_000": 1.5, "var_001": -0.3, "var_002": 2.1}}'
```

Or use a tool like **Postman** or your browser's developer console.

---

## STEP 12: Run ETL Pipeline Separately (Optional, 3 minutes)

If you want to process new data:

```bash
python src/pipeline/etl_pipeline.py \
  --input data/raw/train.csv \
  --output data/processed/train_processed.parquet
```

---

## STEP 13: Run Tests (Optional, 2 minutes)

Verify everything works correctly:

```bash
pytest tests/ -v
```

You should see all tests passing with green checkmarks. ✅

---

## STEP 14: Docker Deployment (Optional, Advanced)

If you have Docker installed:

```bash
# Build and start all services
docker-compose up -d

# Access:
# API: http://localhost:5000
# Dashboard: http://localhost:8501
# MLflow: http://localhost:5001
# Grafana: http://localhost:3000

# Stop all services
docker-compose down
```

---

## 📁 PROJECT STRUCTURE (What Each Folder Contains)

```
banking-prediction/
├── 📂 app/
│   ├── flask/              ← REST API (backend)
│   └── streamlit/          ← Interactive dashboard (frontend)
│
├── 📂 config/
│   ├── config.yaml         ← All settings (paths, thresholds, etc.)
│   └── logging.conf        ← Log format configuration
│
├── 📂 data/
│   ├── raw/                ← Original dataset (train.csv)
│   ├── processed/          ← Cleaned data after ETL
│   └── external/           ← Any reference data
│
├── 📂 docs/
│   ├── 01_business_understanding.md  ← Business case & ROI
│   ├── architecture.md               ← System diagrams
│   └── 14_interview_preparation.md   ← 50 Q&A for interviews
│
├── 📂 models/
│   └── artifacts/          ← Saved trained models
│
├── 📂 reports/
│   ├── figures/            ← All generated charts (PNG)
│   ├── 02_eda_report.md    ← Data analysis report
│   └── technical_report.md ← Full technical documentation
│
├── 📂 src/                 ← All Python source code
│   ├── data/               ← ETL classes (extract, transform, validate)
│   ├── features/           ← Feature engineering & selection
│   ├── models/             ← Training, evaluation, tuning, SHAP
│   └── pipeline/           ← MLOps & versioning
│
├── 📂 tests/               ← Unit tests
│
├── main_pipeline.py        ← Main script - runs everything
├── requirements.txt        ← List of Python packages to install
├── Dockerfile              ← Container definition
└── docker-compose.yml      ← Multi-service orchestration
```

---

## 🛠️ TROUBLESHOOTING

### Problem 1: "python" command not found
**Solution:** Use `python3` instead of `python` (Mac/Linux). On Windows, reinstall Python and check "Add to PATH".

### Problem 2: "pip is not recognized"
**Solution:** 
```bash
python -m ensurepip --upgrade
python -m pip install --upgrade pip
```

### Problem 3: Installation fails with "Microsoft Visual C++ required"
**Solution (Windows):** Download and install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

### Problem 4: "No module named 'xgboost'"
**Solution:** Make sure virtual environment is activated (you see `(venv)`). Then:
```bash
pip install xgboost lightgbm catboost
```

### Problem 5: Port already in use (5000 or 8501)
**Solution:** Kill existing processes:
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Mac/Linux
lsof -ti:5000 | xargs kill -9
```

### Problem 6: Out of memory during training
**Solution:** The dataset is 200MB with 200K rows. Close other applications. If still failing, reduce data:
```python
# In main_pipeline.py, add this after loading data:
df = df.sample(50000, random_state=42)  # Use 50K rows instead
```

### Problem 7: Permission denied (Mac/Linux)
**Solution:**
```bash
chmod +x main_pipeline.py
chmod +x app/flask/app.py
```

---

## 🎯 QUICK COMMANDS CHEAT SHEET

| Task | Command |
|------|---------|
| Activate venv (Windows) | `venv\Scripts\activate` |
| Activate venv (Mac/Linux) | `source venv/bin/activate` |
| Deactivate venv | `deactivate` |
| Run full pipeline | `python main_pipeline.py --phase all --data data/raw/train.csv` |
| Run ETL only | `python src/pipeline/etl_pipeline.py --input data/raw/train.csv --output data/processed/train_processed.parquet` |
| Start API | `python app/flask/app.py` |
| Start Dashboard | `streamlit run app/streamlit/app.py` |
| Run tests | `pytest tests/ -v` |
| Check installed packages | `pip list` |
| Install new package | `pip install package_name` |
| Docker start | `docker-compose up -d` |
| Docker stop | `docker-compose down` |

---

## 📞 NEXT STEPS

After running successfully:

1. **For Interviews:** Read `docs/14_interview_preparation.md` - 50 questions with answers
2. **For Resume:** Copy bullet points from the same file
3. **For Portfolio:** Take screenshots of the dashboard and charts
4. **For Learning:** Modify `main_pipeline.py` to try different models or parameters
5. **For Production:** Deploy using Docker to AWS/Azure/GCP

---

**You're all set! 🎉**

If you get stuck at any step, check the error message carefully. Most issues are:
- Virtual environment not activated (look for `(venv)`)
- Wrong Python version (need 3.9+)
- Missing PATH variables (reinstall Python with "Add to PATH")

Good luck with your project and interviews! 🚀
