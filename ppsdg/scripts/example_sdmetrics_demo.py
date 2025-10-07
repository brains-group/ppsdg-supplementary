from sdmetrics import load_demo
from sdmetrics.reports.single_table import QualityReport

real_data, synthetic_data, metadata = load_demo(modality='single_table')

my_report = QualityReport()
my_report.generate(real_data, synthetic_data, metadata)
