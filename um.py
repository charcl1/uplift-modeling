!pip install econml

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor

"""# Data Loading & Exploration"""

df = pd.read_csv("multi_attribution_sample.csv")
df.head()

df.info()

df.describe()

"""Global Flag, Major Flag, Commercial Flag seem to be heavily imbalanced with low variation so it may not be good to add them to the model.


"""

corr_matrix = df.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr_matrix, annot=True)
plt.show()

"""Interestingly, we see that Global Flag, Major Flag, Commercial Flag, SMC Flag, Employee Count, and PC Count are not highly correlated with size. This suggests that Size isn’t the only meaningful factor, and these other features may help explain additional customer differences.

IT Spend is highly correlated with size so if we add size, it may not be able to add any usefulness about differences between customers.

We should consider only adding either PC Count or Employee Count since they are highy correlated with each other.

"""

clean_cols = ['Revenue', 'IT Spend', 'Employee Count', 'PC Count','Size']
df.hist(
    bins=50,
    figsize=(12, 5)
)
plt.show()

"""# Data Cleaning & Feature Engineering"""

df['Both'] = ((df['Discount'] == 1) & (df['Tech Support'] == 1)).astype(int)

for col in clean_cols:
    df[col] = np.log1p(df[col])

df[clean_cols].hist(bins=50, figsize=(12, 5))
plt.show()

df.describe()

df.dropna(inplace=True)
df.describe()

corr_matrix = df.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr_matrix, annot=True)
plt.show()

"""After logging our correlation matrix didn't change much, likely because the relationships between the variables are nonlinear.

Size, SMC Flag, and Employee Count will be used for the heterogeneity features since they have good variation are not highly correlated with one another.

These features will also be included along with the remaining features as confounders in the model.
"""

def model(df, incentive):

    # Outcome
    Y = df['Revenue']

    # Treatment
    T = df[[incentive]]

    # Heterogeneity feature
    X = df[['Size','SMC Flag','Employee Count']]

    #Confounders
    W = df.drop(columns=['Tech Support', 'Discount', 'Revenue', 'Size','Both'])

    est = CausalForestDML(
        model_y=GradientBoostingRegressor(),
        model_t=GradientBoostingRegressor(),
        discrete_treatment=True,
        n_estimators=1000,
        min_samples_leaf=10,
        max_depth=5,
        random_state=1
    )

    est.fit(Y=Y, T=T, X=X, W=W, cache_values=True)

    print(est.summary())

    cate_preds = est.effect(X)

    df1 = df.copy()
    df1['uplift_decile'] = pd.qcut(cate_preds, 10);

    feats = ['SMC Flag', 'Employee Count', 'Size']

    avg_feat_per_decile = df1.groupby("uplift_decile")[feats[0]].mean().sort_index(ascending=False).reset_index()
    for feat in feats[1:]:
      avg_feat_per_decile[feat] = df1.groupby("uplift_decile")[feat].mean().sort_index(ascending=False).values
    avg_feat_per_decile['group'] = ['Group ' + str(num) for num in range(1,11)]
    avg_feat_per_decile

    fig = plt.figure(figsize=(16,12))
    for idx, feat in enumerate(feats):
      ax = fig.add_subplot(3,3,idx+1)
      ax.set_xlabel('Decile Groups (High Uplift to Low Uplift)')
      ax.set_title(feat)
      ax.plot(avg_feat_per_decile.group, avg_feat_per_decile[feat], 'bx-', linewidth=1, alpha=0.75)
      ax.hlines(df1[feat].mean(), 0, 9, label='Overall Mean', alpha=0.75, linewidth=1.5)
      plt.xticks(rotation='vertical')
      ax.legend()
    plt.tight_layout()
    plt.show()

    return avg_feat_per_decile

"""# Discount Uplift"""

model(df, 'Discount')

"""Small medium corporations with low employee counts and high yearly revenue show the highest predicted uplift, suggesting they are most responsive to discounts.

# Tech Support Uplift
"""

model(df, 'Tech Support')

"""Small medium corporations with low employee count and yearly revenue show the highest predicted uplift, suggesting they are most responsive to tech support.

# Both (Tech Support and Discount) Uplift
"""

model(df, 'Both')

"""Small medium corporations with low employee count and low yearly revenue show the highest predicted uplift for the combined incentives.

This same group also seems to be most responsive to tech support alone, which has a higher uplift than the combined incentive.

Insights & Recomendations:

Best target group for discounts: small medium corporations with low employee count and high yearly revenue

Best target group for tech support: small medium corporations with low employee count and yearly revenue
"""