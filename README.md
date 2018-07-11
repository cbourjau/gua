# Grand unified analysis (GUA)
This is the entry point for the post analysis of the data you get after runnining the the task in AliPhysics/PWGCF/Correlations/C2 on the desired dataset.
You instantiate the `Gua` object (see gua.py) which then offers methods to get the normalized two-particle distributions (r2), and the various Fourier decompositions.
This module depends on `Rootstrap` a package which allows to "bootstrap" uncertainties for very large arrays where ther averages and standard deviations have to be computed on the fly.
`Rootstrap` is a [public repository](https://github.com/cbourjau/rootstrap)
