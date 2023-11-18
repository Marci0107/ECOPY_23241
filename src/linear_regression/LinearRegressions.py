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
        # Calculate standard errors of the coefficients
        X = np.column_stack((np.ones(len(self.right_hand_side)), self.right_hand_side))
        y = self.left_hand_side.values
        n, k = X.shape
        df = n - k  # degrees of freedom
        # Residuals
        residuals = y - X.dot(np.concatenate(([self.alpha], self.beta)))
        # Standard deviation of the residuals
        sigma = np.sqrt((residuals.dot(residuals)) / df)
        # Standard errors of coefficients
        se = np.sqrt(np.diagonal(sigma**2 * np.linalg.inv(X.T.dot(X))))
        # t-statistics
        t_stats = np.concatenate(([self.alpha / se[0]], self.beta / se[1:]))
        # Calculate p-values
        p_values = pd.Series([2 * (1 - stats.t.cdf(np.abs(t), df)) for t in t_stats], name="P-values for the corresponding coefficients")
        self.p_values = p_values
        return p_values
    def get_wald_test_result(self, restriction_matrix):
        # Calculate the Wald test statistic
        wald_value = ((np.array(restriction_matrix).dot(np.concatenate(([self.alpha], self.beta)))) ** 2).sum()
        # Degrees of freedom for the numerator
        df_num = np.array(restriction_matrix).shape[0]
        # Degrees of freedom for the denominator
        df_denom = len(self.right_hand_side) - len(restriction_matrix)
        # Calculate p-value
        p_value = 1 - stats.f.cdf(wald_value, df_num, df_denom)
        # Format the result as a string
        result_string = f"Wald: {wald_value:.3f}, p-value: {p_value:.3f}"
        return result_string

    def get_model_goodness_values(self):
        X = np.column_stack((np.ones(len(self.right_hand_side)), self.right_hand_side))
        y = self.left_hand_side.values
        n, k = X.shape
        df_residuals = n - k  # degrees of freedom for residuals
        df_total = n - 1  # total degrees of freedom
        # Calculate centered R-squared
        y_mean = np.mean(y)
        y_pred = X.dot(np.concatenate(([self.alpha], self.beta)))
        ss_residuals = np.sum((y - y_pred) ** 2)
        ss_total = np.sum((y - y_mean) ** 2)
        centered_r_squared = 1 - (ss_residuals / ss_total)
        # Calculate adjusted R-squared
        adjusted_r_squared = 1 - (ss_residuals / df_residuals) / (ss_total / df_total)
        return f"Centered R-squared: {centered_r_squared:.3f}, Adjusted R-squared: {adjusted_r_squared:.3f}"

class LinearRegressionGLS():
    def __init__(self, df1, df2):
        self.left_hand_side = df1
        self.right_hand_side = df2
        self.V_inv = None # V^(-1)
        self.beta = None  # Coefficients
        self.std_errors = None  # Standard errors

    def fit(self):
        X = np.column_stack((np.ones(len(self.right_hand_side)), np.array(self.right_hand_side)))
        y = np.array(self.left_hand_side)
        # Step 1: OLS estimation
        XTX_inv = np.linalg.inv(X.T @ X)
        beta1 = XTX_inv @ X.T @ y
        # Step 2: Compute residuals
        residuals = y - X @ beta1
        # Step 3: Square residuals
        squared_residuals = residuals ** 2
        # Step 4: GLS estimation using the new model
        beta2 = XTX_inv @ X.T @ np.log(squared_residuals)
        # Step 5: Construct the V inverse matrix
        self.V_inv = np.diag(1 / np.sqrt(np.exp(X @ beta2)))
        # Step 6: GLS estimation using the V inverse matrix
        self.beta = np.linalg.inv(X.T @ self.V_inv @ X) @ X.T @ self.V_inv @ y

    def get_params(self):
        return pd.Series(self.beta, name="Beta coefficients")

    def get_pvalues(self):
        X = np.column_stack((np.ones(len(self.right_hand_side)), np.array(self.right_hand_side)))
        y = np.array(self.left_hand_side)
        xTx = X.T @ self.V_inv @ X
        xTx_inv = np.linalg.inv(xTx)
        # Calculate standard errors
        residuals = y - X @ self.beta
        n, k = X.shape
        df = n - k  # degrees of freedom
        variance = residuals.T @ residuals / df
        self.std_errors = np.sqrt(np.diag(xTx_inv) * variance)
        t_stat = self.beta / self.std_errors
        # Calculate p-values
        term = np.minimum(stats.t.cdf(t_stat, df), 1 - stats.t.cdf(t_stat, df))
        p_values = (term) * 2
        return pd.Series(p_values, name='P-values for the corresponding coefficients')

    def get_wald_test_result(self, restriction_matrix):
        X = np.column_stack((np.ones(len(self.right_hand_side)), np.array(self.right_hand_side)))
        y = np.array(self.left_hand_side)
        xTx = X.T @ self.V_inv @ X
        xTx_inv = np.linalg.inv(xTx)
        n, k = X.shape
        df = n - k  # degrees of freedom
        residuals = y - X @ self.beta
        variance = residuals.T @ residuals / df
        # Calculate Wald statistics
        term1 = restriction_matrix @ self.beta
        term2 = np.linalg.inv(restriction_matrix @ xTx_inv @ np.array(restriction_matrix).T)
        f_stat = (term1.T @ term2 @ term1 / len(restriction_matrix)) / variance
        p_value = 1 - stats.f.cdf(f_stat, len(restriction_matrix), df)
        return f'Wald: {f_stat:.3f}, p-value: {p_value:.3f}'

    def get_model_goodness_values(self):
        y = np.array(self.left_hand_side)
        X = np.column_stack((np.ones(len(self.right_hand_side)), np.array(self.right_hand_side)))
        SSR = y.T @ self.V_inv @ X @ np.linalg.inv(X.T @ self.V_inv @ X) @ X.T @ self.V_inv @ y
        SST = y.T @ self.V_inv @ y
        #Calculate R-squared:
        centered_r_squared = 1 - SSR / SST
        n = len(y)
        p = X.shape[1] - 1
        # Calculate Adjusted R-squared:
        adjusted_r_squared = 1 - (SSR / (n - p - 1)) / (SST / (n - 1))
        return f"Centered R-squared: {centered_r_squared:.3f}, Adjusted R-squared: {adjusted_r_squared:.3f}"