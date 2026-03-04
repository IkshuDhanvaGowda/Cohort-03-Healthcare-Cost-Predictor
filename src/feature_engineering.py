import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_heatmap(df):
    correlation = df.corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix Heatmap")
    plt.show()
