import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('models/vladder_atari/vladder_atari_loss.txt', sep=" ", header=None)
for val in [0, 1, 1, 2, 2, 3, 3, 3, 3]:
	df = df.drop(df.columns[val], axis=1)
df.columns = ["Iteration", "Reconstruction", "Regularization"]
df = df.replace({',':'',':':''}, regex=True)
df = df.apply(pd.to_numeric)
df = df.set_index("Iteration")
for val in ["Reconstruction", "Regularization"]:
	df[val].plot(logy=True)
	plt.title('{} Error'.format(val))
	plt.xlabel('Iterations')
	plt.ylabel('Loss')
	plt.show()
