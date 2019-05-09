The LibreOffice Calc file in this folder contains the quantitative results of the experiments.

The first 5 pages compare the results of subsequent methods across games.
A description of each experiment can be found in the table below.
When the rewards where measured, they were averaged over 10 episodes.
The sheet for each game should have quantative results for each experiment graphed twice.
The largest graph compars every experiment.
The smaller graphs compare every experiment to the vanilla Rainbow.
The 4 transitions experiment ran for 7 days and never got far enough to give any results.

| Column Name | Experiment Description |
| ----------- | ---------------------- |
| Original | Vanilla Rainbow |
| Latent Only | Train on only the latent representation (vanilla VLAE) |
| Latent Concat | Concatenate the latent representation onto the image (vanilla VLAE) |
| Original CC | Add coordinate convolutions to the vanilla rainbow |
| Latent Concat CoordConv | Concatenate the latent representation on the image (VLAE w/ CoordConv)|
| CoordConv Condition | Concatentate the latent representation on the image (CVLAE w/ CoordConv)|
| 4 Transitions | Concatenate rep of 4 images onto 4 images (CVLAE w/ CoordConv)|

We also ran an experiment to see if certain vanilla Rainbow layers were more important to learn then others (with the hope that we can replace those layers with unsupervised learning).
So the "Frozen" sheets show the results of running an experiment to optimization, saving off the weights, and then running a new experiment.
This new experiment loads in the weights for only a certain number of layers and turns off training for those layers.
It then trains the remaining layers to optimality.
The graphs represent the moving averages across 10 games.
I counted optimality as passing the optimal score reported by the Rainbow paper, or for running for 50 million iterations.
I ran out of time, so these sheets are not completed.
