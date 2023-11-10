import statsmodels.api as sm
import numpy as np
import pandas as pd
import scipy.stats as stats
class LinearRegressionSM():

    def __init__(self, df1, df2):
        self.left_hand_side = df1
        self.right_hand_side = df2
        self._model = None

    def fit(self):
        right_df = sm.add_constant(self.right_hand_side)
        model = sm.OLS(self.left_hand_side, right_df).fit()
        self._model = model

    def get_params(self):
        beta_coeffs = self._model.params
        beta_coeffs.name = 'Beta coefficients'
        return beta_coeffs

    def get_pvalues(self):
        p_values = self._model.pvalues
        p_values.name = 'P-values for the corresponding coefficients'
        return p_values

    def get_wald_test_result(self, restr_matrix):
        wald_test = self._model.wald_test(restr_matrix)
        fvalue = wald_test.statistic[0, 0]
        pvalue = wald_test.pvalue
        return f"F-value: {fvalue:.3}, p-value: {pvalue:.3}"

    def get_model_goodness_values(self):
        ars = self._model.rsquared_adj
        ak = self._model.aic
        by = self._model.bic
        return f"Adjusted R-squared: {ars:.3}, Akaike IC: {ak:.3}, Bayes IC: {by:.3}"

class LinearRegressionNP():

    def __init__(self, df1, df2):
        self.left_hand_side = df1
        self.right_hand_side = df2
        self.alpha = None
        self.beta = None
        self.p_values = None

    def fit(self):
        X = np.column_stack((np.ones(len(self.right_hand_side)), self.right_hand_side))
        y = self.left_hand_side.values

        beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        alpha = beta[0]
        beta = beta[1:]
        self.alpha = alpha
        self.beta = beta

    def get_params(self):
        beta_coeffs = pd.Series([self.alpha] + list(self.beta), name="Beta coefficients")
        return beta_coeffs

    def get_pvalues(self):
        n = len(self.left_hand_side)

        # Calculate standard errors of coefficients
        X = np.column_stack((np.ones(n), self.right_hand_side))
        y_hat = X.dot(np.concatenate(([self.alpha], self.beta)))
        residuals = self.left_hand_side - y_hat
        residual_std = np.std(residuals, ddof=1)
        se = np.sqrt(np.linalg.inv(X.T.dot(X)) * residual_std**2)

        # Calculate t-statistics
        t_stats = self.beta / se

        # Calculate p-values
        p_values = pd.Series([min(value, 1 - value) * 2 for value in t_stats], name="P-values for the corresponding coefficients")
        return p_values
    def get_wald_test_result(self, R):
        X = np.column_stack((np.ones(len(self.right_hand_side)), self.right_hand_side))

        dof = len(X) - len(self.beta) - 1

        wald_value = (R.dot(self.beta) / (R.dot(np.linalg.inv(X.T.dot(X)).dot(R.T))))[0]
        p_value = 1 - stats.f.cdf(wald_value, R.shape[0], dof)

        result = f"Wald: {wald_value:.3f}, p-value: {p_value:.3f}"
        return result

    def get_model_goodness_values(self):
        X = np.column_stack((np.ones(len(self.right_hand_side)), self.right_hand_side))
        y = self.left_hand_side.values
        residuals = y - X.dot(np.array([self.alpha] + list(self.beta)))

        y_mean = np.mean(y)
        centered_sum_of_squares = np.sum((y - y_mean) ** 2)
        sum_of_squares_residuals = np.sum(residuals ** 2)

        centered_r_squared = 1 - sum_of_squares_residuals / centered_sum_of_squares
        dof = len(X) - len(self.beta)
        adjusted_r_squared = 1 - (1 - centered_r_squared) * (len(X) - 1) / dof

        result = f"Centered R-squared: {centered_r_squared:.3f}, Adjusted R-squared: {adjusted_r_squared:.3f}"
        return result