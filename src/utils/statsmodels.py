import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
import numpy as np
from statsmodels.api import qqplot
from statsmodels.formula.api import logit
from statsmodels.graphics.mosaicplot import mosaic


# Draw the scatter plot
sns.scatterplot(x="n_convenience",
                y="price_twd_msq",
                data=taiwan_real_estate)
# Show the plot
plt.show()


# linear regression
mdl_price_vs_conv = ols("price_twd_msq ~ n_convenience",
                        data=taiwan_real_estate)
# Fit the model
mdl_price_vs_conv = mdl_price_vs_conv.fit()
print(mdl_price_vs_conv.params)


# Histograms of price_twd_msq with 10 bins, split by the age of each house
sns.displot(data=taiwan_real_estate,
         x="price_twd_msq",
         col="house_age_years",
         bins=10)
# Show the plot
plt.show()


# Calculate the mean of price_twd_msq, grouped by house age
mean_price_by_age = taiwan_real_estate.groupby("house_age_years")["price_twd_msq"].mean()
print(mean_price_by_age)


# Create the model, fit it
mdl_price_vs_age = ols("price_twd_msq~house_age_years", data=taiwan_real_estate).fit()
# Print the parameters of the fitted model
print(mdl_price_vs_age.params)


# Create explanatory_data
explanatory_data = pd.DataFrame({'n_convenience': np.arange(0, 11)})
# Use mdl_price_vs_conv to predict with explanatory_data, call it price_twd_msq
price_twd_msq = mdl_price_vs_conv.predict(explanatory_data)
# Create prediction_data
prediction_data = explanatory_data.assign(
    price_twd_msq = price_twd_msq)
# Print the result
print(prediction_data)


# Create a new figure, fig
fig = plt.figure()
#regplot
sns.regplot(x="n_convenience",
            y="price_twd_msq",
            data=taiwan_real_estate,
            ci=None)
# Add a scatter plot layer to the regplot
sns.scatterplot(x="n_convenience",
               y="price_twd_msq",
               data=prediction_data,
               color = "red")
# Show the layered plot
plt.show()


# Define a DataFrame impossible
impossible = pd.DataFrame({"n_convenience": [-1, 2.5]})
price_twd_msq = mdl_price_vs_conv.predict(impossible)
# Create prediction_data
prediction_data = impossible.assign(
    price_twd_msq = price_twd_msq)
# Print the result
print(prediction_data)


# Get the coefficients of mdl_price_vs_conv
coeffs = mdl_price_vs_conv.params
# Get the intercept
intercept = coeffs[0]
# Get the slope
slope = coeffs[1]
# Manually calculate the predictions
price_twd_msq = intercept + slope * explanatory_data
print(price_twd_msq)
# Compare to the results from .predict()
print(price_twd_msq.assign(predictions_auto=mdl_price_vs_conv.predict(explanatory_data)))


# Print the coeff of determination for mdl_click_vs_impression_orig
print(mdl_click_vs_impression_orig.rsquared)


# Print a summary of mdl_click_vs_impression_orig
print(mdl_click_vs_impression_orig.summary())


# Calculate mse_orig for mdl_click_vs_impression_orig
mse_orig = mdl_click_vs_impression_orig.mse_resid
# Calculate rse_orig for mdl_click_vs_impression_orig and print it
rse_orig = mse_orig ** 0.5
print("RSE of original model: ", rse_orig)
# Calculate mse_trans for mdl_click_vs_impression_trans
mse_trans = mdl_click_vs_impression_trans.mse_resid
# Calculate rse_trans for mdl_click_vs_impression_trans and print it
rse_trans = mse_trans ** 0.5
print("RSE of transformed model: ", rse_trans)


# Plot the residuals vs. fitted values
sns.residplot(x="n_convenience", y="price_twd_msq", data=taiwan_real_estate, lowess=True)
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.show()


# Q-Q Plot
qqplot(data=mdl_price_vs_conv.resid, fit=True, line="45")
plt.show()


# Scale-location plot
model_norm_residuals = mdl_price_vs_conv.get_influence().resid_studentized_internal
model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))
# Create the scale-location plot
sns.regplot(x=mdl_price_vs_conv.fittedvalues, y=model_norm_residuals_abs_sqrt, ci=None, lowess=True)
plt.xlabel("Fitted values")
plt.ylabel("Sqrt of abs val of stdized residuals")
plt.show()


# Create summary_info
summary_info = mdl_price_vs_dist.get_influence().summary_frame()

# Add the hat_diag column to taiwan_real_estate, name it leverage
taiwan_real_estate["leverage"] = summary_info["hat_diag"]
# Sort taiwan_real_estate by leverage in descending order and print the head
print(taiwan_real_estate.sort_values(by="leverage", ascending=False).head())

# Add the cooks_d column to taiwan_real_estate, name it cooks_dist
taiwan_real_estate["cooks_dist"] = summary_info["cooks_d"]
# Sort taiwan_real_estate by cooks_dist in descending order and print the head.
print(taiwan_real_estate.sort_values(by="cooks_dist", ascending=False).head())


# Draw a linear regression trend line and a scatter plot of time_since_first_purchase vs. has_churned
sns.regplot(x="time_since_first_purchase",
            y="has_churned",
            data=churn,
            ci=None,
            line_kws={"color": "red"})
# Draw a logistic regression trend line and a scatter plot of time_since_first_purchase vs. has_churned
sns.regplot(x="time_since_first_purchase",
            y="has_churned",
            data=churn,
            ci=None,
            logistic=True,
            line_kws={"color": "blue"})
plt.show()


# Fit a logistic regression of churn vs. length of relationship using the churn dataset
mdl_churn_vs_relationship = logit("has_churned ~ time_since_first_purchase", data=churn).fit()
# Print the parameters of the fitted model
print(mdl_churn_vs_relationship.params)


# Update prediction data by adding most_likely_outcome
prediction_data["most_likely_outcome"] = np.round(prediction_data["has_churned"])
print(prediction_data.head())


# Update prediction data with odds_ratio
prediction_data["odds_ratio"] = prediction_data["has_churned"] / (1 - prediction_data["has_churned"])


# Update prediction data with log_odds_ratio
prediction_data["log_odds_ratio"] = np.log(prediction_data["odds_ratio"])
print(prediction_data.head())


# Get the actual responses
actual_response = churn["has_churned"]
# Get the predicted responses
predicted_response = np.round(mdl_churn_vs_relationship.predict())
# Create outcomes as a DataFrame of both Series
outcomes = pd.DataFrame({"actual_response": actual_response,
                         "predicted_response": predicted_response})
# Print the outcomes
print(outcomes.value_counts(sort = False))


# Calculate the confusion matrix conf_matrix
conf_matrix = mdl_churn_vs_relationship.pred_table()
# Draw a mosaic plot of conf_matrix
mosaic(conf_matrix)
plt.show()


# Extract TN, TP, FN and FP from conf_matrix
TN = conf_matrix[0, 0]
TP = conf_matrix[1, 1]
FN = conf_matrix[1, 0]
FP = conf_matrix[0, 1]
# Calculate and print the accuracy
accuracy = (TN + TP) / (TN + TP + FN + FP)
print("accuracy: ", accuracy)
# Calculate and print the sensitivity
sensitivity = TP / (FN + TP)
print("sensitivity: ", sensitivity)
# Calculate and print the specificity
specificity = TN / (TN + FP)
print("specificity: ", specificity)