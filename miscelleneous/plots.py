import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd

df=pd.read_csv('data_by_genres_o.csv')

scatter=sns.scatterplot(data=df, x='valence', y='energy', hue='genres')
print(df['genres'].dtype)
df['genres'] = df['genres'].astype('category')

plt.legend(loc='best', fontsize='x-small')
legend = scatter.legend_
legend.get_frame().set_facecolor('white')
plt.figure(figsize=(10,10))
plt.show()
result = ('graphed',)