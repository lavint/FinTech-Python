# PyViz Installation Guide

PyViz is a Python visualization package that provides a single platform to access multiple visualization packages, including Matplotlib, Plotly Express, hvPlot, Panel, D3.js, etc.


1. Deactivate your current Conda environment. This is required in order to update the global Conda environment. Be sure to quit any running applications such as Jupyter prior to deactivating the environment.

```
conda deactivate
```

2. Update Conda.

```
conda update conda
```

3. Install the PyViz dependencies.

```
conda install -c conda-forge nodejs
conda install -c pyviz hvplot
conda install -c plotly plotly
conda install -c pyviz panel
```

4. Install the Jupyter Lab Extensions.

```
jupyter labextension install @pyviz/jupyterlab_pyviz
jupyter labextension install @jupyterlab/plotly-extension
```

5. Resolve `ModuleNotFoundError: No module named 'prompt_toolkit.enums'`
```
pip uninstall prompt-toolkit
pip install prompt-toolkit

pip install -U jupyter_console   # only if the above doesn't work
```

Check Jupyter version
```
jupyter --version
```


Upgrade IPython Notebook
```
pip install -U jupyter
```

Downgrade ipykernal
```
pip install 'ipykernel<5.2.0'
```