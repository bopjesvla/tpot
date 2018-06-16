import pandas as pd

df = pd.read_csv('10gen10pop.csv').set_index('name')

s = [c for c in df.columns if 'score' in c]

scores = df[s]

ranks = df[s].rank(axis=1, ascending=False)

print(ranks.mean())
