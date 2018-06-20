import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np

df = pd.read_csv('10gen10pop.csv').set_index('name')

s = [c for c in df.columns if 'score' in c]

df['good_tpot'] = df['xgb_score'] < df['normal_score']
df['good_ensemble'] = df['normal_score'] < df['ensemble_score']

N = df['instances']

scores = df[s]

print(s)

sign_test = pd.concat([(~scores.le(scores[column], axis=0)).sum(axis=0) for column in s], axis=1)
sign_test.columns = s
print(sign_test)

sign_test.to_latex('signtest.tex')

ranks = df[s].rank(axis=1, ascending=False)

ranks.mean().to_latex('ranks.tex')

print(ranks.mean())

print(stats.spearmanr(N, df['good_tpot']))
print(stats.spearmanr(N, df['good_ensemble']))

groups = np.floor(np.log10(df['instances']))
by_instances = df.groupby(groups)
groupN = by_instances.apply(lambda x: len(x))
good_tpotN = by_instances.apply(lambda x: sum(x['good_tpot']) / len(x))
good_ensembleN = by_instances.apply(lambda x: sum(x['good_ensemble']) / len(x))
N = by_instances.apply(lambda x: str(int(10 ** x.name)) + ' - ' + str(int(10 ** (x.name+1) - 1)))

tableN = pd.concat([N, groupN, good_tpotN, good_ensembleN], axis=1)
tableN.columns = ['Sample size', 'N', 'TPOT > XGB', 'Ensemble > TPOT']
tableN.set_index('Sample size', inplace=True)
tableN.to_latex('N.tex')

print(tableN)
