import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil, sqrt

def prepare_plot_fig(plots, n_cols = None, d_type = 'num', title = None):
    """Prepare the figure for plotting"""
    if n_cols is None:
        ncols = min(5, ceil(sqrt(len(plots))))
    else:
        ncols = n_cols
    nrows = ceil(len(plots) / ncols)

    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3), layout='constrained')
    if type(axs) == np.ndarray:
        axs = axs.flatten()
    if title is None:
        title = 'Numerical Columns' if d_type == 'num' else 'Categorical & Binary Columns'
    fig.suptitle(title, fontsize=max(ncols * 4, 16), fontweight='bold')

    return nrows, ncols, fig, axs

def plot_distributions(df, bins = 30, kde = True, hue = None, n_cols = None):
    """Plot the distributions of numerical and categorical columns"""

    # Separate columns by type
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    num_cols = df.select_dtypes(include='number').columns
    for col in num_cols:
        if len(df[col].unique()) == 2:
            cat_cols = cat_cols.append(pd.Index([col]))
            num_cols = num_cols.drop(col)
    
    # Plot numerical columns
    if len(num_cols) > 0:
        nrows, ncols, fig, axs = prepare_plot_fig(num_cols, d_type='num', n_cols=n_cols)

        for i, col in enumerate(num_cols, start=1):
            plt.subplot(nrows, ncols, i)
            ax = sns.histplot(x=col, data=df, bins=bins, kde=kde, hue=hue if col != hue else None)
            height = max([p.get_height() for p in ax.patches]) * 1.05
            ax.set_ylim(0,height)
            ax.set(ylabel='', xlabel='')
            plt.title(col, fontsize=max(ncols * 2, 10), fontweight='bold')
        plt.show()

    # Plot categorical columns
    if len(cat_cols) > 0:
        nrows, ncols, fig, axs = prepare_plot_fig(cat_cols, d_type='cat', n_cols=n_cols)

        for i, col in enumerate(cat_cols, start=1):
            plt.subplot(nrows, ncols, i)
            ax = sns.countplot(x=col, data=df, hue=hue if col != hue else None)
            ax.set(ylabel='', xlabel='')
            plt.title(col, fontsize=max(ncols * 2, 10), fontweight='bold')
        plt.show()

def boxplot_grid(df, hue = None, n_cols = None):
    """Plot the boxplots of numerical columns"""

    cols = df.select_dtypes(include='number').columns.drop(hue, errors='ignore')
    nrows, ncols, fig, axs = prepare_plot_fig(cols, n_cols=n_cols)

    for i, col in enumerate(cols, start=1):
        plt.subplot(nrows, ncols, i)
        ax = sns.boxplot(y=col, x=hue, data=df)
        ax.set(ylabel='', xlabel='')
        plt.title(col, fontsize=max(ncols * 2, 10), fontweight='bold')
    plt.show()

def scatterplot_high_corr(df, threshold = 0.5, hue = None, n_cols = None, **kwargs):
    """Plot the scatterplots of highly correlated numerical columns"""

    cols = df.select_dtypes(include='number').columns.drop(hue, errors='ignore')

    plots = []
    for x in cols:
        for y in cols:
            if x != y:
                pearson = df[x].corr(df[y])
                if abs(pearson) > threshold:
                    plots.append((x, y, pearson))
        cols = cols.drop(x)

    title = f'Highly Correlated Columns (Pearson > {threshold})'
    nrows, ncols, fig, axs = prepare_plot_fig(plots, n_cols=n_cols, title=title)
    for i, (x, y, p) in enumerate(plots, start=1):
        plt.subplot(nrows, ncols, i)
        sns.scatterplot(x=x, y=y, data=df, hue=hue, **kwargs)
        plt.title(f'{x.upper()} vs {y.upper()} ({p :.2f})', fontsize=max(ncols * 2, 10), fontweight='bold')