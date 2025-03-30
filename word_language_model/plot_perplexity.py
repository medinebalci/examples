import pandas as pd
import matplotlib.pyplot as plt

# Load data
log_file = "C:/Users/medin/OneDrive/Dokumente/mt-exercise-02/scripts/tools/pytorch-examples/word_language_model/perplexities.log"
data = pd.read_csv(log_file, sep='\t', header=None)

# Assign proper column names
data.columns = ["Epoch", "Train_Perplexity", "Validation_Perplexity"]

# Plot
plt.figure(figsize=(10,5))
plt.plot(data["Epoch"], data["Train_Perplexity"], label="Train Perplexity", marker="o")
plt.plot(data["Epoch"], data["Validation_Perplexity"], label="Validation Perplexity", marker="s")
plt.xlabel("Epochs")
plt.ylabel("Perplexity")
plt.legend()
plt.title("Training vs Validation Perplexity")
plt.grid(True)
plt.show()
