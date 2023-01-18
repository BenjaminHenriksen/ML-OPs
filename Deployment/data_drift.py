import pandas as pd
from sklearn import datasets
import numpy as np
reference_data = datasets.load_iris(as_frame='auto').frame
current_data = pd.read_csv('prediction_database.csv')

print(current_data.head())
"""
Standardize the dataframes such that they have the same column names and drop the time column from the current_data dataframe.
"""
current_data = current_data.drop(columns=['time'])
# convert reference_data to a dataframe
reference_data = pd.DataFrame(reference_data)

print(current_data.head())
print(reference_data.head())
# rename the columns of reference_data
reference_data.columns = current_data.columns

print(current_data.head())
print(reference_data.head())

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference_data, current_data=current_data)
report.save_html('report.html')

## write some nan values to the reference data
for i in range(15):
    random_instance = np.random.randint(0, reference_data.shape[0])
    random_column = np.random.randint(0, reference_data.shape[1])
    reference_data.iloc[random_instance, random_column] = np.nan

from evidently.metric_preset import DataDriftPreset, DataQualityPreset
report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
report.run(reference_data=reference_data, current_data=current_data)
report.save_html('report_DQ.html')

from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
report = Report(metrics=[DataDriftPreset(), DataQualityPreset(), TargetDriftPreset()])
report.run(reference_data=reference_data, current_data=current_data)
report.save_html('report_target_drift.html')


from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfMissingValues
data_test = TestSuite(tests=[TestNumberOfMissingValues()])
data_test.run(reference_data=reference_data, current_data=current_data)
data_test.save_html('data_test.html')