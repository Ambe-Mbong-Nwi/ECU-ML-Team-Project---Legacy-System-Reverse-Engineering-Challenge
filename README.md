# Legacy Reimbursement System — Machine Learning Reverse Engineering

**Project Type:** Team Project  
**Group Number:** **5**

### Team Members

- Jayath Premasinghe  
- Ambe Mbong-Nwi Nchang  

---

## Project Overview

This project reverse-engineers a long-running black-box travel reimbursement system whose internal logic is undocumented and opaque.  
The legacy system contains hard-coded business rules, non-linear logic, and deterministic artifacts that make it difficult to replace using a single mathematical formula.

Using:

- Historical reimbursement data  
- Employee interview insights  
- Machine learning and rule-based modeling  

the objective is to:

- Discover hidden reimbursement logic  
- Replicate the behavior of the legacy system  
- Implement a reliable, interpretable, and production-ready replacement model  

---

## Problem Definition

### Inputs

The system accepts exactly three parameters:

- `trip_duration_days` — integer  
- `miles_traveled` — float  
- `total_receipts_amount` — float  

### Output

- A single reimbursement amount (USD), rounded to two decimal places  

### Evaluation Criteria

- Exact match: ± $0.01  
- Close match: ± $1.00  
- Score: combined accuracy and precision (lower is better)  

---

## Repository Structure and Project Approach

The repository is organized to reflect a phased analytical workflow, progressing from data exploration to final system integration.

---

### Exploratory Data Analysis and Feature Engineering

#### `data_exploration_report.ipynb`

This notebook performs:

- Statistical summaries of raw input and output variables  
- Distribution analysis and outlier detection  
- Correlation analysis with the reimbursement amount  
- Missing data assessment  
- Translation of interview insights into engineered features  

A total of 24 domain-specific features are created, including ratios, interaction terms, polynomial features, and rule-based flags.  
This notebook establishes the business logic hypotheses that guide all subsequent modeling.

---

### Model Evaluation by Machine Learning Category

Each required machine learning category is evaluated independently using a dedicated notebook.

#### `linear_regression_variants.ipynb`

- Simple Linear Regression  
- Ridge Regression  
- Lasso Regression  
- Polynomial Regression  

Demonstrates underfitting and inability to model threshold-driven behavior.

---

#### `tree_based_methods.ipynb`

- Decision Trees  
- Random Forest  
- Gradient Boosting  
- XGBoost  
- LightGBM  

Shows strong performance due to the ability to model piecewise, rule-like logic.

---

#### `advance_techniques.ipynb`

- Support Vector Regression  
- Neural Networks (MLP)  
- Ensemble methods (stacking and voting)  

Evaluates advanced techniques and highlights their limitations in deterministic systems.

---

#### `rule_based_learning.ipynb`

- Explicit rule extraction  
- Deterministic baseline logic  
- Validation of interview-derived rules  

Provides the structural backbone for the final hybrid approach.

---

### Final Model and Results

#### `solution.ipynb`

This notebook integrates all prior analysis and implements the final model:

- Rule-based baseline reimbursement logic  
- Residual learning using Gradient Boosting and XGBoost  
- Model blending to reduce variance  
- Isotonic calibration to remove systematic bias  

Final evaluation metrics and comparison tables are produced here.

---

### Technical Report

#### `technical_report.qmd`

A comprehensive technical report covering:

- Problem formulation  
- Data exploration  
- Feature engineering strategy  
- Model evaluation and comparison  
- Final model selection  
- Business insights and recommendations  

---

## Data Files

- `public_cases.json`  
  Raw dataset of 1,000 labeled reimbursement cases  

- `public_cases_derived_features.csv`  
  Dataset containing original inputs plus 24 engineered features  

---

## System Integration and Execution

### `reimbursement_calculator.py`

This file implements the production inference pipeline, including:

1. Rule-based baseline computation  
2. Feature construction  
3. Residual machine learning prediction  
4. Isotonic calibration  
5. Deterministic rounding  

This script is invoked exclusively through `run.sh`, in accordance with the required execution contract.

---

### Running the Model (Required Interface)

The solution must be executed using `run.sh`.

```bash
chmod +x run.sh
./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>
