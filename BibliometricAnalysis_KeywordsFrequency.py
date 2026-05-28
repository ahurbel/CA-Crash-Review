import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# Load the dataset
#df = pd.read_csv("ALL_Bibliometric_Records.csv")
df = pd.read_csv("Selected_Records.csv", encoding="latin1")

# Extract and clean keywords
all_keywords = df['Author Keywords'].dropna().apply(lambda x: [kw.strip().lower() for kw in x.split(';')])
flat_keywords = [kw for sublist in all_keywords for kw in sublist]

# Count frequencies
keyword_counts = Counter(flat_keywords)
top_keywords = keyword_counts.most_common(20)  # Top 20 keywords

# Convert to DataFrame for plotting
df_top_keywords = pd.DataFrame(top_keywords, columns=["Keyword", "Frequency"])

# Plot
plt.figure(figsize=(10, 7))
plt.barh(df_top_keywords["Keyword"][::-1], df_top_keywords["Frequency"][::-1], color="black")

# Set fonts and colors manually
plt.xlabel("Frequency", fontname="Arial", color="black", fontsize=14)
#plt.title("Top 20 Most Frequent Author Keywords", fontname="Arial", color="black", fontsize=14)
plt.yticks(fontname="Arial", color="black", fontsize=14)
plt.xticks(fontname="Arial", color="black", fontsize=14)

plt.tight_layout()
plt.savefig("Frequency_Words.pdf", format="pdf")
plt.show()
