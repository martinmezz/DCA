import matplotlib.pyplot as plt

def rate_vs_cumulative(time, rate, cumulative):
    """Plot Rate vs Cumulative Production."""
    plt.figure(figsize=(8, 6))
    plt.plot(cumulative, rate, marker='o')
    plt.xlabel('Cumulative Production (bbl)')
    plt.ylabel('Rate (bbl/day)')
    plt.title('Rate vs Cumulative Production')
    plt.savefig('output/rate_cumulative_plot.png')
    plt.show()

def diagnostic_function_plot(time, rate):
    """Plot Diagnostic Function (Rate vs Time in Log-Log Scale)."""
    plt.figure(figsize=(8, 6))
    plt.loglog(time, rate, marker='o')
    plt.xlabel('Time (days)')
    plt.ylabel('Rate (bbl/day)')
    plt.title('Diagnostic Function Plot (Log-Log)')
    plt.savefig('output/diagnostic_function_plot.png')
    plt.show()
