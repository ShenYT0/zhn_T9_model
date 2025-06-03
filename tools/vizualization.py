import matplotlib.pyplot as plt
from collections import Counter

def plot_length_distribution(lengths, title="Length Distribution", xlabel="Length", ylabel="Frequency"):
    length_counts = Counter(lengths)
    lengths_sorted = sorted(length_counts.items())
    x_vals, y_vals = zip(*lengths_sorted)

    plt.figure(figsize=(10, 6))
    plt.bar(x_vals, y_vals)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Example usage:
# plot_length_distribution(sentence_lengths, title="Sentence Length Distribution (Characters)")
# plot_length_distribution(digit_lengths, title="Digit Sequence Length Distribution")