# Uplift Modeling
This project explored a b2b tech startup that sells software and wants to understand the effects of discounts, tech support, and their combination on revenue across different customer types without running randomized experiments.

The goal was to identify the customer group that responds best to each incentive to maximize ROI.

The data used is from Kaggle and contained information about customer features, which incentive each customer received, and how much revenue the customer generated after receiving the incentives.

## What I Did
**Exploration** - Explored the variables and their distributions

**Cleaning**  - Handled outliers and missing values 

**Feature Engineering** - Created a new interaction feature

**Modeling** - Estimated CATE using Causal Forest DML 

**Interpretation/Visualization** - Summarized uplift deciles and identified high-response segments

## Insights & Recommendations
**Best target group for discounts:** Small medium corporations with low employee count and high yearly revenue

**Best target group for tech support:** Small medium corporations with low employee count and yearly revenue
