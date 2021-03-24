import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_excel('DSM_excel.xlsx')
DSM = np.array(df)
print(DSM)

plt.matshow(DSM)
plt.show()

