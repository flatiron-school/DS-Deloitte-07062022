import pandas as pd
df = pd.read_csv('game-of-thrones-deaths-data.csv')
print(df.groupby('season')['character_killed'].count())