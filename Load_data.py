import csv
import numpy as np


# Download data from https://archive.ics.uci.edu/ml/datasets/spambase
FILE_NAME = "spambase.data"

# 1) load with csv file
with open(FILE_NAME, "r") as f:
    data = list(csv.reader(f, delimiter=","))

data = np.array(data, dtype=np.float32)
print(data.shape)

# 2) load with np.loadtxt()
# skiprows=1
data = np.loadtxt(FILE_NAME, delimiter=",", dtype=np.float32)
print(data.shape, data.dtype)

# 3) load with np.genfromtxt()
# skip_header=0, missing_values="---", filling_values=0.0
data = np.genfromtxt(FILE_NAME, delimiter=",", dtype=np.float32)
print(data.shape)