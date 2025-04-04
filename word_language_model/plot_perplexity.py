import csv
import matplotlib.pyplot as plt
import pandas as pd


# Log data for each dropout rate (valid perplexity values only)
dropout_0 = [
    399.2531, 196.3306, 155.6438, 134.5034, 128.8710, 
    122.2894, 124.2278, 125.1296, 126.4028, 126.9243, 
    126.9367, 126.9290, 126.9267, 126.9261, 126.9259,
    126.9259, 126.9260, 126.9259, 126.9259, 126.9259,
    126.9259, 126.9259, 126.9259, 126.9259, 126.9259,
    126.9259, 126.9259, 126.9259, 126.9259, 126.9259,
    126.9259, 126.9259, 126.9259, 126.9259, 126.9259,
    126.9259, 126.9259, 126.9259, 126.9259, 126.9259,
]

dropout_03 = [
    301.6173, 200.2271, 157.5760, 138.0620, 127.2805, 
    118.9713, 115.9849, 114.1332, 117.1944, 114.2709, 
    114.9664, 115.1490, 115.2220, 115.2315, 115.2319,
    115.2323, 115.2322, 115.2322, 115.2322, 115.2322,
    115.2322, 115.2322, 115.2322, 115.2322, 115.2322,
    115.2322, 115.2322, 115.2322, 115.2322, 115.2322,
    115.2322, 115.2322, 115.2322, 115.2322, 115.2322,
    115.2322, 115.2322, 115.2322, 115.2322, 115.2322,
]

dropout_06 = [
    306.7431, 204.3154, 161.6963, 143.4286, 131.8059, 
    127.0321, 121.1911, 115.9094, 112.8455, 112.5643, 
    111.7557, 110.4456, 110.3101, 108.5075, 108.3916,
    109.0043, 107.8663, 108.4943, 109.0475, 108.9844,
    108.9049, 108.9149, 108.9149, 108.9144, 108.9145,
    108.9146, 108.9146, 108.9146, 108.9146, 108.9146,
    108.9146, 108.9146, 108.9146, 108.9146, 108.9146,
    108.9146, 108.9146, 108.9146, 108.9146, 108.9146,
]

dropout_08 = [
    323.2149, 229.4185, 191.9283, 170.1954, 160.8057, 
    155.5123, 148.6022, 143.8130, 142.0072, 141.5837, 
    138.8798, 134.7818, 132.6061, 132.9871, 130.2359,
    129.3949, 128.3105, 127.7125, 126.8411, 125.6944,
    125.0863, 124.6830, 124.5050, 124.2602, 124.1228,
    124.0400, 124.3463, 124.0564, 124.4847, 124.5314,
    124.5250, 124.5238, 124.5242, 124.5241, 124.5241,
    124.5241, 124.5241, 124.5241, 124.5241, 124.5241,
]

# Combine the data
epochs = [i for i in range(1, 41)]  # Total 40 epochs
data = list(zip(epochs, dropout_0, dropout_03, dropout_06, dropout_08))

# Writing data into a CSV file
with open('perplexity_log.csv', 'w', newline='') as csvfile:
    fieldnames = ['Epoch', 'Dropout 0.0', 'Dropout 0.3', 'Dropout 0.6', 'Dropout 0.8']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for epoch, drop0, drop03, drop06, drop08 in data:
        writer.writerow({
            'Epoch': epoch, 
            'Dropout 0.0': drop0, 
            'Dropout 0.3': drop03, 
            'Dropout 0.6': drop06, 
            'Dropout 0.8': drop08
        })


# Data from logs
data = {
    "Epoch": list(range(1, 41)),
    "Dropout 0.0": [5.9896, 5.2798, 5.0476, 4.9016, 4.8588, 4.8064, 4.8221, 4.8294, 4.8395, 4.8436] + [4.8436]*30,
    "Dropout 0.3": [5.7092, 5.2995, 5.0599, 4.9277, 4.8464, 4.7789, 4.7535, 4.7374, 4.7638, 4.7386] + [4.7469]*30,
    "Dropout 0.6": [5.7260, 5.3197, 5.0857, 4.9658, 4.8813, 4.8444, 4.7974, 4.7528, 4.7260, 4.7235] + [4.6906]*30,
    "Dropout 0.8": [5.7783, 5.4355, 5.2571, 5.1369, 5.0802, 5.0467, 5.0013, 4.9685, 4.9559, 4.9529] + [4.8245]*30
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Save training perplexity to CSV
df.to_csv("training_perplexity.csv", index=False)

# Test perplexity dictionary
test_perplexity = {
    "Dropout": [0.0, 0.3, 0.6, 0.8],
    "Test Perplexity": [4.59, 4.50, 4.39, 4.55]
}
df_test = pd.DataFrame(test_perplexity)
df_test.to_csv("test_perplexity.csv", index=False)

# Plot Training Perplexity
plt.figure(figsize=(10, 5))
for col in df.columns[1:]:
    plt.plot(df["Epoch"], df[col], label=col)

plt.xlabel("Epoch")
plt.ylabel("Training Perplexity")
plt.title("Training Perplexity Over Epochs")
plt.legend()
plt.grid()
plt.savefig("training_perplexity_plot.png")
plt.show()


# Load data
log_file = "C:/Users/medin/OneDrive/Dokumente/mt-exercise-02/scripts/tools/pytorch-examples/word_language_model/perplexities.log"
data = pd.read_csv(log_file, sep='\t', header=None)

# Assign proper column names
data.columns = ["Epoch", "Train_Perplexity", "Validation_Perplexity"]

# Plot Train Perplexity and Validation Perplexity
plt.figure(figsize=(10,5))
plt.plot(data["Epoch"], data["Train_Perplexity"], label="Train Perplexity", marker="o")
plt.plot(data["Epoch"], data["Validation_Perplexity"], label="Validation Perplexity", marker="s")
plt.xlabel("Epochs")
plt.ylabel("Perplexity")
plt.legend()
plt.title("Training vs Validation Perplexity")
plt.grid(True)
plt.savefig("test_validation_perplexity_plot.png")
plt.show()
