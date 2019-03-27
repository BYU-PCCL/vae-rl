import matplotlib.pyplot as plt
import sys
import numpy as np
plt.style.use('seaborn')

fpath = sys.argv[1]
plot_name = sys.argv[2]
chop = int(sys.argv[3])

with open(fpath, 'r') as f:
	data = f.readlines()

reconstruction = [float(line.split()[4][:-1]) for line in data if len(line.split()) > 0]
regularization = [float(line.split()[7][:-1]) for line in data if len(line.split()) > 0]

epoch = str(data[-1].split()[1][:-1])

reconstruction, regularization = reconstruction[chop:], regularization[chop:]

plt.plot(np.arange(len(reconstruction))*20, reconstruction)
plt.title("Reconstruction Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.savefig(plot_name+epoch+'_reconloss.pdf')
plt.show()

plt.plot(np.arange(len(regularization))*20, regularization)
plt.title("Regularization Loss")
plt.ylabel("Loss")
plt.xlabel("Iteration")
plt.savefig(plot_name+epoch+'_regloss.pdf')
plt.show()
