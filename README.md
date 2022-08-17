# CPCA_Python
Here we will constantly release the Python version of CPCA code.
# user guide
First, input the data of Z and G matrix by inputting column numbers of your dataset into Z_col and G_col.

Then you will get a scree plot, select number of components based on the plot and input into the window below. When it reaches 100% it means it is completed and then you can proceed to the next step.

After that, following sorts of graph will show up:
- colored heatmaps, illustrating component loadings and predictor loadings, larger the loading, deeper the color
- heatmaps in black and white, which only show significant loadings
- line graphs of realiability test, you need to identify and input the number of variance below according to the graph.
then you will obtain a final conncection diagram, for which the width of connected lines correspond to size of loadings; the predictor loadings are on the left and the component loadings are on the right.
