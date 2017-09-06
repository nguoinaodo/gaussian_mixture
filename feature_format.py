import csv
import numpy as np 

# Read datasets
file = open('./dataset/sale.csv', 'r')
result = csv.reader(file, delimiter = ',', quotechar = '"') 
data = []
for row in result:
	data.append(row)
data = np.array(data)

# Crop and convert type
data = data[1:, 1: 5].astype(np.float)

# Normalization
cols_max = np.max(data, axis = 0)
cols_min = np.min(data, axis = 0)
data = (data - cols_min) / cols_max

# Train test split 
N = data.shape[0]
Ntr = 2 * N / 3
train = data[: Ntr]
test = data[Ntr:]
