import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr

# ======================= BOXPLOT =============================
def plot_boxplot(
    df, 
    vars, 
    target, 
    top_n=None, 
    figsize_width=12, 
    figsize_height=None
):
    """
    Plot boxplots for one or multiple variables against a target variable.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the data.
    vars : list
        List of variables to plot on the x-axis.
    target : str
        Target variable to plot on the y-axis.
    top_n : int or None, default=None
        If specified, only the top N most frequent categories will be plotted.
    figsize_width : int, default=12
        Width of the figure.
    figsize_height : int or None, default=None
        Height of the figure. If None, it is set to 4 times the number of variables.
    """

    # Dynamic height if not specified
    if figsize_height is None:
        figsize_height = 4 * len(vars)

    # Remove rows with NaN in any of the selected variables or the target
    df_plot = df[vars + [target]].dropna()

    # Create subplots
    fig, axes = plt.subplots(nrows=len(vars), ncols=1, figsize=(figsize_width, figsize_height))

    # Ensure axes is iterable when only one variable is plotted
    if len(vars) == 1:
        axes = [axes]

    for i, var in enumerate(vars):

        # Filtrar top_n categorías o valores más frecuentes
        if top_n is not None:
            top_values = (df_plot[var].value_counts().nlargest(top_n).index)
            df_plot_top = df_plot[df_plot[var].isin(top_values)]
        else:
            df_plot_top = df_plot.copy()

        # Detect variable type
        if pd.api.types.is_numeric_dtype(df_plot_top[var]):
            # Numeric → natural order
            order = sorted(df_plot_top[var].unique())
        else:
            # Categorical → order by median of target
            order = (
                df_plot_top.groupby(var, observed=False)[target]
                .median()
                .sort_values(ascending=False)
                .index
            )

        # Create boxplot
        sns.boxplot(x=df_plot_top[var], y=df_plot_top[target], ax=axes[i], order=order)
        axes[i].set_title(f"{target} by {var}")
        axes[i].set_xlabel(var)
        axes[i].set_ylabel(target)
    
        # Get current ticks and labels
        ticks = axes[i].get_xticks()
        labels = [lbl.get_text() for lbl in axes[i].get_xticklabels()]

        # Calculate maximum label length
        max_label_len = max(len(lbl) for lbl in labels if lbl)

        # Shorten long labels
        new_labels = [lbl[:12] + "..." if len(lbl) > 12 else lbl for lbl in labels]

        # Set ticks and labels
        axes[i].set_xticks(ticks)
        axes[i].set_xticklabels(new_labels)

        # Always rotate labels
        for lbl in axes[i].get_xticklabels():
            lbl.set_rotation(45)
            # Align to the right only if labels are long
            if max_label_len > 8:
                lbl.set_horizontalalignment('right')

    plt.tight_layout()
    plt.show()


# ======================= BARPLOT =============================
def plot_barplot(
    df, 
    vars, 
    top_n=None, 
    figsize_width=12, 
    figsize_height=None
):
    """
    Plot barplots for one or multiple variables showing the distribution of categories.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the data.
    vars : list
        List of variables to plot on the x-axis.
    top_n : int or None, default=None
        If specified, only the top N most frequent categories will be plotted.
    figsize_width : int, default=12
        Width of the figure.
    figsize_height : int or None, default=None
        Height of the figure. If None, it is set to 4 times the number of variables.
    """

    # Dynamic height if not specified
    if figsize_height is None:
        figsize_height = 4 * len(vars)

    # Remove rows with NaN in any of the selected variables
    df_plot = df[vars].dropna()

    # Create subplots
    fig, axes = plt.subplots(nrows=len(vars), ncols=1, figsize=(figsize_width, figsize_height))

    # Ensure axes is iterable when only one variable is plotted
    if len(vars) == 1:
        axes = [axes]

    for i, var in enumerate(vars):

        # Conditional selection of top values (only most frequent categories)
        if top_n is not None:
            counts = df_plot[var].value_counts().nlargest(top_n)
        else:
            counts = df_plot[var].value_counts()

        # Plot barplot with filtered counts
        axes[i].bar(counts.index, counts.values)

        # Set titles and labels
        axes[i].set_title(f"Distribution of {var}")
        axes[i].set_xlabel(var)
        axes[i].set_ylabel("Count")

        # Get current labels
        ticks = axes[i].get_xticks()
        labels = [str(lbl) for lbl in counts.index]

        # Calculate maximum label length
        max_label_len = max(len(lbl) for lbl in labels if lbl)

        # Shorten long labels
        new_labels = [lbl[:12] + "..." if len(lbl) > 12 else lbl for lbl in labels]

        # Set ticks and labels
        axes[i].set_xticks(range(len(new_labels)))
        axes[i].set_xticklabels(new_labels)

        # Rotate labels only if they are long and align to the right
        if max_label_len > 8:
            for lbl in axes[i].get_xticklabels():
                lbl.set_rotation(45)
                lbl.set_horizontalalignment('right')  # align end of label with the bar

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()


# ======================= HISTOGRAM PLOT =============================
def plot_histogram(
    df, 
    vars, 
    figsize_width=12, 
    figsize_height=None,
    bins=50, 
    kde=False,
    binrange=None
):
    """
    Plot histograms for one or multiple variables.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the data.
    vars : list
        List of variables to plot.
    figsize_width : int, default=12
        Width of the figure.
    figsize_height : int or None, default=None
        Height of the figure. If None, it is set to 4 times the number of variables.
    bins : int, default=50
        Number of bins for the histogram.
    binrange : tuple, optional
        Value range for the histogram (default=None → full data range).
    kde : bool, default=False
        Whether to include a Kernel Density Estimate curve.
    """

    # Dynamic height if not specified
    if figsize_height is None:
        figsize_height = 4 * len(vars)

    # Remove rows with NaN in any of the selected variables
    df_plot = df[vars].dropna()

    # Create subplots
    fig, axes = plt.subplots(nrows=len(vars), ncols=1, figsize=(figsize_width, figsize_height))

    # Ensure axes is iterable when only one variable is plotted
    if len(vars) == 1:
        axes = [axes]
    
    # Set histogram 
    if binrange is None:
        binrange = (df_plot[vars].min().min(), df_plot[vars].max().max())

    for i, var in enumerate(vars):
        
        # Plot histogram for each variable
        sns.histplot(df_plot[var], bins=bins, binrange=binrange, kde=False, ax=axes[i])

        axes[i].set_title(f"Distribution of {var}")
        axes[i].set_xlabel(var)
        axes[i].set_ylabel("Count")

        # Apply x-axis limit conditionally per subplot
        if (df_plot[var] < 0).any():
            # If variable has negative values → do nothing
            pass
        else:
            # If variable has no negatives → force x-axis to start at 0
            axes[i].set_xlim(left=0)

        # Get current ticks and labels
        ticks = axes[i].get_xticks()
        labels = [lbl.get_text() for lbl in axes[i].get_xticklabels()]

        # Calculate maximum label length
        max_label_len = max(len(lbl) for lbl in labels if lbl)

        # Shorten long labels
        new_labels = [lbl[:12] + "..." if len(lbl) > 12 else lbl for lbl in labels]

        # Set ticks and labels
        axes[i].set_xticks(ticks)
        axes[i].set_xticklabels(new_labels)

        # Rotate labels only if they are long and align to the right
        if max_label_len > 8:
            for lbl in axes[i].get_xticklabels():
                lbl.set_rotation(45)
                lbl.set_horizontalalignment('right')  # align end of label with the bar

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()


# ======================= REGPLOT =============================
def plot_regplot(
    df, 
    vars, 
    target,
    figsize_width=10, 
    figsize_height=None,
    alpha=0.3
):
    """
    Plot regression plots (scatter plots with regression line) for one or multiple variables
    against a target variable, including Pearson correlation coefficients.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the data.
    vars : list
        List of independent variables to plot against the target.
    target : str
        The dependent variable (column name) to be plotted on the y-axis.
    figsize_width : int, default=12
        Width of the figure.
    figsize_height : int or None, default=None
        Height of the figure. If None, it is set to 4 times the number of variables.
    alpha : float, default=0.3
        Transparency level for scatter points.
    """

    # Dynamic height if not specified
    if figsize_height is None:
        figsize_height = 4 * len(vars)

    # Remove rows with NaN in any of the selected variables or the target
    df_plot = df[vars + [target]].dropna()

    # Create subplots
    fig, axes = plt.subplots(nrows=len(vars), ncols=1, figsize=(figsize_width, figsize_height))

    # Ensure axes is iterable when only one variable is plotted
    if len(vars) == 1:
        axes = [axes]
        
    for i, var in enumerate(vars):
        # Calculate Pearson correlation coefficient
        corr, pval = pearsonr(df_plot[var], df_plot[target])
        
        # Plot scatter with regression line
        sns.regplot(x=df_plot[var], y=df_plot[target], ax=axes[i],
                    scatter_kws={'alpha':alpha}, line_kws={'color':'red'})
        
        # Title with correlation coefficient
        axes[i].set_title(f"{target} vs {var} (Pearson r = {corr:.2f})")
        axes[i].set_xlabel(var)
        axes[i].set_ylabel(target)

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()
