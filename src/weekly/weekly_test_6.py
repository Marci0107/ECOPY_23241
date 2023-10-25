import pandas as pd
import statsmodels
from statsmodels.formula.api import ols

sp_500_df = pd.read_parquet("C:/Users/User\Documents\GitHub\ECOPY_23241\data\sp500.parquet", engine='fastparquet')
ff_factors_df = pd.read_parquet("C:/Users/User\Documents\GitHub\ECOPY_23241\data/ff_factors.parquet", engine='fastparquet')

merged_df = pd.merge(sp_500_df, ff_factors_df, on='Date', how='left')

merged_df['Excess Return'] = merged_df['Monthly Returns'] - merged_df['Mkt-RF']

sorted_df = merged_df.sort_values(by='Date')

sorted_df['ex_ret_1'] = sorted_df.groupby('Symbol')['Excess Return'].shift(-1)
sorted_df.sort_values(by=['Symbol', 'Date'])

cleared_df = sorted_df.dropna(subset=['ex_ret_1'])
cleared_df.dropna(subset=['HML'])

cleared_df[cleared_df["Symbol"] == 'AMZN']

final_df = cleared_df.drop(columns=['Symbol'])

class LinearRegressionSM():

    def __init__(self, df1, df2):
        self.left_hand_side = df1
        self.right_hand_side = df2

    def fit(self):
        right_df = sm.add_constant(self.right_hand_side)
        self._model = ols("self.left_hand_side~right_df").fit()

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
        return f"F-value: {fvalue}, p-value: {pvalue}"

    def get_model_goodness_values(self):
        ars = round(self._model.rsquared_adj, 3)
        ak = round(self._model.aic, 3)
        by = round(self._model.bic, 3)
        return f"Adjusted R-squared: {ars}, Akaike IC: {ak}, Bayes IC: {by}"