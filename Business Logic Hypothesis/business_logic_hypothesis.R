---
  title: "Business Logic Hypothesis Report"
subtitle: "Legacy Reimbursement System Reverse Engineering"
author: "<Your Name>"
date: "`r format(Sys.Date())`"
format:
  html:
  toc: true
toc-depth: 3
number-sections: true
---
  
  ## 1. Purpose
  
  This document outlines hypotheses about the underlying business logic in the legacy travel reimbursement system.  
It is based on:
  
  - Interview transcripts
- Product requirements
- Observed legacy system outputs (where available)
- Domain knowledge provided by finance & travel policy stakeholders

Goal: **Reconstruct likely logic and non-linear patterns to replicate legacy behavior**.

---
  
  ## 2. System Inputs and Outputs
  
  ### Inputs
  
  | Variable | Description |
  |---|---|
  Days | Number of travel days |
  Miles | Total miles traveled |
  Receipt Total | Dollar value of submitted receipts |
  User/Employee Profile | Possible long-term behavior tracking |
  Date Submitted | Potential calendar influence |
  
  ### Output
  
  | Variable | Description |
  |---|---|
  Reimbursement Amount | Single numeric output |
  
  ---
  
  ## 3. Known Legacy System Requirements
  
  - Single output with no breakdown
- Uses days, miles, receipts at minimum
- Behavior must be replicated **including quirks/bugs**
  - Not fully documented; evolved over time

---
  
  ## 4. Hypothesized Business Rules
  
  ### 4.1 Per-Diem Logic
  
  | Hypothesis | Evidence Source | Notes |
  |---|---|---|
  Base per diem ~\$100/day | Accounting | Consistent baseline |
  5-day trips receive bonus | Accounting + Sales | Sometimes inconsistent |
  Very short / very long trips penalized | Multiple interviews | Encourages mid-length travel |
  
  ---
  
  ### 4.2 Mileage Logic
  
  | Hypothesis | Evidence | Notes |
  |---|---|---|
  ~$0.58/mile for first 100 miles | Accounting | Cap then taper |
  Diminishing rate after threshold | Multiple interviews | Curve likely |
  Sweet spot: 180–220 mi/day | Procurement | Efficiency bonus zone |
  Penalty >400 mi/day | Sales & Procurement | Perceived non-business travel |
  
  ---
  
  ### 4.3 Receipt Logic
  
  | Hypothesis | Evidence | Notes |
  |---|---|---|
  Optimal spend \$600–\$800 | Accounting | Peak benefit |
  Low spend sometimes penalized | Marketing | Inconsistent |
  High spend diminishing return | Accounting | Vacation penalty on long trips |
  Small receipts penalized (<$40) | Multiple interviews | Anti-gaming behavior |
  
  ---
  
  ### 4.4 Efficiency / Multi-Factor Logic
  
  | Hypothesis | Evidence | Notes |
  |---|---|---|
  Miles/Day matters | Procurement | Efficient travel rewarded |
  Spend/Day matters by trip length | Accounting & Procurement | Tiered thresholds |
  Trip type clusters exist | Procurement clustering | ~6 behavioral groups |
  
  ---
  
  ### 4.5 Temporal Effects
  
  | Hypothesis | Evidence | Certainty |
  |---|---|---|
  End of quarter bump | Sales + others | Medium |
  Day of week effect (Tues > Fri) | Procurement | Low–Medium |
  Possible monthly cycle | Sales | Low |
  Submission timing matters | HR | Medium |
  
  ---
  
  ### 4.6 User Behavior Memory
  
  | Hypothesis | Evidence | Notes |
  |---|---|---|
  Past spending behavior affects future reimbursements | Sales | Could be profile scoring |
  New employees get lower reimbursements | HR | Could also be behavior learning curve |
  
  ---
  
  ## 5. Suspected Bugs / Anomalies
  
  | Bug Hypothesis | Description | Evidence |
  |---|---|---|
  Rounding oddities | .49 or .99 receipts favorable | Accounting |
  Magic total rumor (\$847) | “Lucky number” payout anecdote | Sales |
  Random noise 5–10% | Variation even with similar trips | Several users |
  
  ---
  
  ## 6. Derived Feature Plan
  
  | Feature | Purpose |
  |---|---|
  Miles per day | Efficiency score |
  Spend per day | Behavior ratio |
  Spend per mile | Efficiency spending |
  Receipt rounding indicator (.49/.99) | Rounding bug modeling |
  Quarter / Month / Weekday | Calendar effects |
  Employee historical spend behavior | User memory theory |
  Trip category flags | Multi-path rules |
  
  ---
  
  ## 7. Validation Strategy
  
  We will test hypotheses via:
  
  - Correlation & non-linear regression analysis
- Tree-based model feature importance (RF, XGBoost)
- Partial dependence plots
- Threshold break analysis (kinks at 5 days, 100 miles etc.)
- Clustering for rule-path detection

---
  
  ## 8. Open Questions
  
  | Question | Next Steps |
  |---|---|
  Is randomness intentional or emergent? | Residual noise analysis |
  Does user history impact results? | Compare cohorts over time |
  How many calculation paths? | Clustering + tree model splits |
  
  ---
  
  ## 9. Summary
  
  The legacy system likely consists of:
  
  - Per-diem base with bonuses/penalties
- Mileage diminishing returns curve
- Spending efficiency curve & trip-length interaction
- Temporal & user history adjustments
- Multiple internal rule paths
- Noise/randomness to prevent gaming
- Long-standing rounding quirks

These findings guide feature engineering and model replication strategy.

---
  