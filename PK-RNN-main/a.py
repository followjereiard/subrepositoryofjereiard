import pickle
import pandas as pd
import numpy as np

with open(file='sample_data.pkl', mode='rb') as f:    
    data = pickle.load(f)
print(type(data))

for i, d in enumerate(data):
    df = pd.DataFrame(d)
    df.to_csv(f"sample_data_{i}.csv")