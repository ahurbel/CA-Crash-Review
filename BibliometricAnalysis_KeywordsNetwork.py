# 📚 Imports
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from community import community_louvain
import itertools

# 📂 Load your dataset
df = pd.read_csv("Selected_Records.csv", encoding="latin1")

# ================================================
# 🔗 1. Keyword Co-occurrence Network
# ================================================
keyword_graph = nx.Graph()
df_keywords = df['Author Keywords'].dropna().apply(lambda x: [kw.strip().lower() for kw in x.split(';') if kw.strip()])

for keywords in df_keywords:
    for kw1, kw2 in itertools.combinations(sorted(set(keywords)), 2):
        if keyword_graph.has_edge(kw1, kw2):
            keyword_graph[kw1][kw2]['weight'] += 1
        else:
            keyword_graph.add_edge(kw1, kw2, weight=1)

# ================================================
# 👥 2. Co-authorship Network
# ================================================
author_graph = nx.Graph()
df_authors = df['Authors'].dropna().apply(lambda x: [a.strip() for a in x.split(';') if a.strip()])

for authors in df_authors:
    for a1, a2 in itertools.combinations(sorted(set(authors)), 2):
        if author_graph.has_edge(a1, a2):
            author_graph[a1][a2]['weight'] += 1
        else:
            author_graph.add_edge(a1, a2, weight=1)

# ================================================
# 📄 3. Document Similarity Network (TF-IDF + Cosine)
# ================================================
df['FullText'] = df['Title'].fillna('') + " " + df['Abstract'].fillna('')
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['FullText'])
cos_sim = cosine_similarity(tfidf_matrix)

doc_graph = nx.Graph()
titles = df['Title'].fillna('').tolist()
for i in range(len(titles)):
    for j in range(i + 1, len(titles)):
        if cos_sim[i, j] > 0.3:  # threshold for linking
            doc_graph.add_edge(titles[i], titles[j], weight=cos_sim[i, j])

# ================================================
# 🧠 4. Cluster Detection with Louvain Algorithm
# ================================================
def detect_communities(graph, label):
    if len(graph.nodes) == 0:
        print(f"No nodes in {label} network.")
        return {}
    partition = community_louvain.best_partition(graph)
    nx.set_node_attributes(graph, partition, 'community')
    print(f"{label}: {len(set(partition.values()))} communities detected.")
    return partition

keyword_partition = detect_communities(keyword_graph, "Keyword Co-occurrence")
author_partition = detect_communities(author_graph, "Co-authorship")
doc_partition = detect_communities(doc_graph, "Document Similarity")

# ✅ Optional: Export graphs for Gephi or visualization
nx.write_gexf(keyword_graph, "keyword_network.gexf")
nx.write_gexf(author_graph, "author_network.gexf")
nx.write_gexf(doc_graph, "document_similarity_network.gexf")


def plot_network(graph, partition, title, max_nodes=100, top_labels=25):
    if len(graph.nodes) > max_nodes:
        print(f"Too many nodes to display ({len(graph.nodes)}), showing top {max_nodes} by degree.")
        top_nodes = sorted(graph.degree, key=lambda x: x[1], reverse=True)[:max_nodes]
        graph = graph.subgraph([n for n, _ in top_nodes])
        partition = {k: v for k, v in partition.items() if k in graph.nodes}

    pos = nx.spring_layout(graph, seed=42)
    cmap = plt.get_cmap('tab20')
    node_color = [cmap(partition.get(n, 0) % 20) for n in graph.nodes]
    node_size = [graph.degree(n) * 1000 for n in graph.nodes]
    edge_alpha = [graph[u][v]['weight'] / max(1, max(nx.get_edge_attributes(graph, 'weight').values())) for u, v in graph.edges]

    plt.figure(figsize=(10, 7))
    nx.draw_networkx_edges(graph, pos, alpha=edge_alpha, width=3.0)
    nx.draw_networkx_nodes(graph, pos, node_size=node_size, node_color=node_color, alpha=0.9)

    # ✅ Only label top nodes by degree
    top_nodes = sorted(graph.degree, key=lambda x: x[1], reverse=True)[:top_labels]
    label_dict = {n: n for n, _ in top_nodes}
    nx.draw_networkx_labels(graph, pos, labels=label_dict, font_size=14, font_family='Arial')

    #plt.title(title, fontname="Arial")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("KeyWordsNetwork.pdf", format="pdf")
    plt.show()



plot_network(keyword_graph, keyword_partition,
             title="Keyword Co-occurrence Network",
             max_nodes=20, top_labels=30)




