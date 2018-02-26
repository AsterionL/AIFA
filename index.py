import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

index =pd.read_csv('index.csv')
index = index.drop(['TD'],axis=1)
print(index.corr())



plt.figure();
index.plot();
plt.show()