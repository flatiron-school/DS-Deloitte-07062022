import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('game-of-thrones-deaths-data.csv')
print(df.groupby('season')['character_killed'].count().plot(kind='bar'))

plt.show()
