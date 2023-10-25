from statsmodels.formula.api import ols
import statsmodels.api as sm
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
        fvalue = round(wald_test.statistic, 3)
        pvalue = round(wald_test.pvalue, 3)
        return f"F-value: {fvalue:.3}, p-value: {pvalue:.3}"

    def get_model_goodness_values(self):
        ars = round(self._model.rsquared_adj, 3)
        ak = round(self._model.aic, 3)
        by = round(self._model.bic, 3)
        return f"Adjusted R-squared: {ars:.3}, Akaike IC: {ak:.3}, Bayes IC: {by:.3}"