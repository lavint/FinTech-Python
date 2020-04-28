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

4. Avoid "JavaScript heap out of memory" errors during extension installation
    ```
    # (OS X/Linux)
    export NODE_OPTIONS=--max-old-space-size=4096

    # (Windows)
    set NODE_OPTIONS=--max-old-space-size=4096
    ```

4. Install the Jupyter Lab Extensions.

    ```
    jupyter labextension install @jupyter-widgets/jupyterlab-manager@1.1 --no-build

    jupyter labextension install jupyterlab-plotly@4.6.0 --no-build
    
    jupyter labextension install plotlywidget@4.6.0 --no-build

    jupyter labextension install @pyviz/jupyterlab_pyviz --no-build

    ```

5. Build the extensions (This may take a few minutes)

    ```shell
    jupyter lab build
    ```

6. After the build, unset the node options that you used above.

    ```shell
    # Unset NODE_OPTIONS environment variable
    # (OS X/Linux)
    unset NODE_OPTIONS

    # (Windows)
    set NODE_OPTIONS=
    ```

7. Check packages
    ```
    conda list | grep pyviz
    conda list | grep plotly
    jupyter labextension list
    ```


<br>
<br>

**Troubleshooting**


Resolve `ModuleNotFoundError: No module named 'prompt_toolkit.enums'`
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



Uninstall an extension
```
jupyter labextension uninstall my-extension
```



Update all extensions
```
jupyter labextension update --all
```



Check jupyter lab extension list
```
jupyter labextension list
```


Fix blank jupyter lab
```
pip uninstall jupyterlab
pip install jupyterlab
```