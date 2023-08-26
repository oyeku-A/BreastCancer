import pandas as pd

data_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
cols = ['Sample_code_number', 'Clump_thickness', 'Uniformity_of_cell_size', 'Uniformity_of_cell_shape', 'Marginal_adhesion', 'Single_epithelial_cell_size', 'Bare_nuclei', 'Bland_chromatin', 'Normal_nucleoli', 'Mitoses', 'Class']
dataset = pd.read_csv(data_path, names=cols, index_col='Sample_code_number')

dataset['Class'] = dataset['Class'].map(lambda x: 1 if x == 2 else 0)
dataset['Bare_nuclei'] = pd.to_numeric(dataset['Bare_nuclei'], errors='coerce')

dataset.to_csv('cleaned.csv')
