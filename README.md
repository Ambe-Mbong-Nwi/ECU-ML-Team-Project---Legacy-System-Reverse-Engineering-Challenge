# ECU Machine Learning Team Project
## Legacy-System-Reverse-Engineering-Challenge

## Team Members
- Jayath Premasinghe
- Ambe Mbong-Nwi Nchang

## Goal
Our team will reverse-engineer a 60-year-old travel reimbursement system using only historical data and employee interviews, applying machine learning techniques to discover hidden business logic patterns and create predictive models that replicate legacy system behavior.

## Scenario
Our team has been hired as ML consultants by ACME Corporation. Their legacy reimbursement system has been running for 60 years, no one knows how it works, but it is still used daily. A new system has been built, but the ACME Corporation is confused by the differences in results. Our mission is to use machine learning to understand the original business logic and create a model that can explain and predict the legacy system’s behavior.


## Project Specification
### Problem Statement

#### Input Variables (provided by the legacy system):

- trip_duration_days: Number of days spent traveling (integer)

- miles_traveled: Total miles traveled (integer)

- total_receipts_amount: Total dollar amount of receipts (float)

#### Output Variable (to predict):

- Single numeric reimbursement amount (float, rounded to 2 decimal places)

#### Success Criteria:

- Exact matches: Cases within ±$0.01±$0.01 of the expected output

- Close matches: Cases within ±$1.00±$1.00 of the expected output

- Score: Lower is better (combines accuracy and precision)
- 

### Dataset Description

1,000 historical input/output examples from public_cases.json is available. Create a random sample of 750 examples for use in training the machine learning models. The remaining 250 examples should be used for testing.

#### Additional Resources:

- Product Requirements Document (PRD) with business context (available at https://github.com/8090-inc/top-coder-challenge/blob/main/PRD.md).

- Employee interview transcripts with system behavior hints (available at https://github.com/8090-inc/top-coder-challenge/blob/main/INTERVIEWS.md).

- Domain knowledge about travel reimbursement policies.

