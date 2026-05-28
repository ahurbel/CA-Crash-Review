import pandas as pd
import matplotlib.pyplot as plt

# Load the bibliometric dataset with correct encoding
df = pd.read_csv("ALL_Bibliometric_RecordsALL.csv", encoding="latin1")
# Alternative encodings if needed:
# df = pd.read_csv("ALL_Bibliometric_Records.csv", encoding="ISO-8859-1")
# df = pd.read_csv("ALL_Bibliometric_Records.csv", encoding="cp1252")

# Ensure year is integer (drop missing or malformed entries)
df = df[pd.to_numeric(df['Year'], errors='coerce').notnull()]
df['Year'] = df['Year'].astype(int)

# Group by year and count publications
pubs_per_year = df['Year'].value_counts().sort_index()

# Plotting
plt.figure(figsize=(10, 7))
bars = plt.bar(pubs_per_year.index.astype(str), pubs_per_year.values,
               color='black', edgecolor='black')

# Titles and labels
#plt.title("Publications per Year", fontname="Arial", fontsize=14)
plt.xlabel("Year", fontname="Arial", fontsize=14)
plt.ylabel("Number of Publications", fontname="Arial", fontsize=14)
plt.xticks(rotation=45, fontname="Arial", fontsize=12)
plt.yticks(fontname="Arial", fontsize=12)

# Force y-axis to use integers only
plt.gca().yaxis.get_major_locator().set_params(integer=True)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("Time_TrendALL.pdf", format="pdf")
plt.show()
