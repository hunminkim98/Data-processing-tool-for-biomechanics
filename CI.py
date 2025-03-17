import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt

def calculate_confidence_interval(data, confidence_level=0.95, method='normal'):
    """
    Calculate confidence interval for the given data.
    
    Parameters:
    -----------
    data : array-like
        Input data for which to compute the confidence interval
    confidence_level : float, optional (default=0.95)
        Confidence level (between 0 and 1)
    method : str, optional (default='normal')
        Method to calculate confidence interval:
        - 'normal': Assumes data is normally distributed
        - 'bootstrap': Non-parametric bootstrap method
        - 't': Uses t-distribution (better for small samples)
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'mean': sample mean
        - 'lower_bound': lower bound of confidence interval
        - 'upper_bound': upper bound of confidence interval
        - 'confidence_level': the confidence level used
        - 'std': sample standard deviation
    """
    # Convert input to numpy array for consistent handling
    data = np.array(data)
    
    # Remove NaN values
    data = data[~np.isnan(data)]
    
    # Calculate basic statistics
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)  # ddof=1 for sample standard deviation
    
    if n == 0:
        raise ValueError("No valid data points provided")
    
    if method == 'normal':
        # Z-value for the given confidence level
        z = stats.norm.ppf((1 + confidence_level) / 2)
        # Standard error of the mean
        se = std / np.sqrt(n)
        # Calculate confidence interval
        margin_of_error = z * se
        lower_bound = mean - margin_of_error
        upper_bound = mean + margin_of_error
    
    elif method == 't':
        # T-value for the given confidence level and degrees of freedom
        t = stats.t.ppf((1 + confidence_level) / 2, df=n-1)
        # Standard error of the mean
        se = std / np.sqrt(n)
        # Calculate confidence interval
        margin_of_error = t * se
        lower_bound = mean - margin_of_error
        upper_bound = mean + margin_of_error
    
    elif method == 'bootstrap':
        # Number of bootstrap samples
        n_bootstrap = 10000
        # Generate bootstrap samples
        bootstrap_means = np.random.choice(data, size=(n_bootstrap, n), replace=True).mean(axis=1)
        # Calculate confidence interval from bootstrap distribution
        lower_percentile = (1 - confidence_level) / 2 * 100
        upper_percentile = (1 + confidence_level) / 2 * 100
        lower_bound = np.percentile(bootstrap_means, lower_percentile)
        upper_bound = np.percentile(bootstrap_means, upper_percentile)
    
    else:
        raise ValueError("Method must be one of 'normal', 't', or 'bootstrap'")
    
    return {
        'mean': mean,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'confidence_level': confidence_level,
        'std': std,
        'n': n
    }

def plot_confidence_interval(data, confidence_level=0.95, method='normal', title=None, xlabel=None, ylabel=None):
    """
    Calculate and plot confidence interval for the given data.
    
    Parameters:
    -----------
    data : array-like or dict of array-like
        Input data for which to compute the confidence interval.
        If dict, each key-value pair will be plotted as a separate group.
    confidence_level : float, optional (default=0.95)
        Confidence level (between 0 and 1)
    method : str, optional (default='normal')
        Method to calculate confidence interval: 'normal', 'bootstrap', or 't'
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    
    Returns:
    --------
    tuple
        (fig, ax) matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if isinstance(data, dict):
        # Multiple groups
        x_positions = np.arange(len(data))
        labels = list(data.keys())
        
        means = []
        errors = []
        ci_results = []
        
        for group_data in data.values():
            ci = calculate_confidence_interval(group_data, confidence_level, method)
            means.append(ci['mean'])
            errors.append([ci['mean'] - ci['lower_bound'], ci['upper_bound'] - ci['mean']])
            ci_results.append(ci)
        
        # Plot bars with error bars
        ax.bar(x_positions, means, yerr=np.array(errors).T, capsize=10, alpha=0.7)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels)
        
        # Add text annotations
        for i, ci in enumerate(ci_results):
            ax.text(x_positions[i], means[i] * 0.9, 
                   f"Mean: {ci['mean']:.2f}\nCI: [{ci['lower_bound']:.2f}, {ci['upper_bound']:.2f}]",
                   ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))
    
    else:
        # Single group
        ci = calculate_confidence_interval(data, confidence_level, method)
        
        # Plot mean with error bars
        ax.bar([0], [ci['mean']], yerr=[[ci['mean'] - ci['lower_bound']], [ci['upper_bound'] - ci['mean']]], 
               capsize=10, alpha=0.7, width=0.5)
        ax.set_xticks([0])
        ax.set_xticklabels(['Data'])
        
        # Add text annotation
        ax.text(0, ci['mean'] * 0.9, 
               f"Mean: {ci['mean']:.2f}\nCI: [{ci['lower_bound']:.2f}, {ci['upper_bound']:.2f}]",
               ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))
    
    # Add title and labels
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    
    ax.set_title(f"{confidence_level*100:.0f}% Confidence Intervals ({method} method)")
    plt.tight_layout()
    
    return fig, ax

# Example usage:
if __name__ == "__main__":
    # Sample data
    sample_data = np.random.normal(loc=100, scale=15, size=30)
    
    # Calculate confidence interval
    ci_normal = calculate_confidence_interval(sample_data, confidence_level=0.95, method='normal')
    ci_t = calculate_confidence_interval(sample_data, confidence_level=0.95, method='t')
    ci_bootstrap = calculate_confidence_interval(sample_data, confidence_level=0.95, method='bootstrap')
    
    # Print results
    print("Normal distribution method:")
    print(f"Mean: {ci_normal['mean']:.2f}")
    print(f"95% CI: [{ci_normal['lower_bound']:.2f}, {ci_normal['upper_bound']:.2f}]")
    
    print("\nt-distribution method:")
    print(f"Mean: {ci_t['mean']:.2f}")
    print(f"95% CI: [{ci_t['lower_bound']:.2f}, {ci_t['upper_bound']:.2f}]")
    
    print("\nBootstrap method:")
    print(f"Mean: {ci_bootstrap['mean']:.2f}")
    print(f"95% CI: [{ci_bootstrap['lower_bound']:.2f}, {ci_bootstrap['upper_bound']:.2f}]")
    
    # Plot confidence intervals
    plot_confidence_interval(sample_data)
    plt.show()
    
    # Example with multiple groups
    group_data = {
        'Group A': np.random.normal(loc=100, scale=15, size=30),
        'Group B': np.random.normal(loc=90, scale=10, size=25),
        'Group C': np.random.normal(loc=110, scale=20, size=40)
    }
    
    plot_confidence_interval(group_data, confidence_level=0.95, method='t',
                           title="Comparison Between Groups",
                           xlabel="Groups", ylabel="Value")
    plt.show()