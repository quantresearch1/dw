import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
from community import community_louvain


def create_bipartite_graph(matrix: npt.ArrayLike) -> nx.Graph:
    """
    Create a bipartite graph from a sparse matrix.
    Rows and columns are represented as nodes, and non-zero entries as edges.
    """
    G = nx.Graph()

    # Add row nodes (prefix with 'r' to distinguish from column nodes)
    for i in range(matrix.shape[0]):
        G.add_node(f"r{i+1}", bipartite=0)  # +1 to match 1-indexed example

    # Add column nodes (prefix with 'c')
    for j in range(matrix.shape[1]):
        G.add_node(f"c{j+1}", bipartite=1)  # +1 to match 1-indexed example

    # Add edges for non-zero entries
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] != 0:
                G.add_edge(f"r{i+1}", f"c{j+1}")

    return G


def visualize_bipartite_graph(G: nx.Graph, title: str = "Bipartite Graph") -> None:
    """
    Visualize the bipartite graph.
    """
    plt.figure(figsize=(10, 8))

    # Separate row and column nodes
    row_nodes: list[str] = [n for n, d in G.nodes(data=True) if d.get("bipartite") == 0]
    col_nodes: list[str] = [n for n, d in G.nodes(data=True) if d.get("bipartite") == 1]

    # Create positions
    pos: dict[str, tuple[int, int]] = {}
    pos.update((node, (1, i)) for i, node in enumerate(row_nodes))
    pos.update((node, (2, i)) for i, node in enumerate(col_nodes))

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=row_nodes, node_color="lightblue")
    nx.draw_networkx_nodes(G, pos, nodelist=col_nodes, node_color="lightgreen")

    # Draw edges
    nx.draw_networkx_edges(G, pos)

    # Draw labels
    nx.draw_networkx_labels(G, pos)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def partition_graph(G: nx.Graph) -> dict[str, int]:
    """
    Partition the graph using community detection.
    For simplicity, we'll use the Louvain method from the community library.
    """
    # Convert bipartite graph to a projected graph for community detection
    partition: dict[str, int] = community_louvain.best_partition(G, random_state=3)
    return partition


def reorder_matrix(
    matrix: npt.ArrayLike, partition: dict[str, int]
) -> tuple[npt.ArrayLike, list[int], list[int]]:
    """
    Reorder the matrix according to the partition.
    """
    # Extract row and column indices and their community assignments
    row_communities: dict[int, int] = {}
    col_communities: dict[int, int] = {}

    for node, community in partition.items():
        if node.startswith("r"):
            row_idx = int(node[1:]) - 1  # Convert back to 0-indexed
            row_communities[row_idx] = community
        elif node.startswith("c"):
            col_idx = int(node[1:]) - 1  # Convert back to 0-indexed
            col_communities[col_idx] = community

    # Sort rows and columns by community
    row_order = sorted(row_communities.keys(), key=lambda k: (row_communities[k], k))
    col_order = sorted(col_communities.keys(), key=lambda k: (col_communities[k], k))

    # Create mapping from old to new indices
    row_mapping = {old: new for new, old in enumerate(row_order)}
    col_mapping = {old: new for new, old in enumerate(col_order)}

    # Create reordered matrix
    reordered = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] != 0:
                reordered[row_mapping[i], col_mapping[j]] = matrix[i, j]

    return reordered, row_order, col_order


def identify_blocks(
    matrix: npt.ArrayLike, partition: dict[str, int]
) -> tuple[npt.ArrayLike, list[int], list[int]]:
    """
    Identify the blocks in the reordered matrix.
    """
    # TODO: Implement more sophisticated block identification
    # For now, we'll just return the reordered matrix
    reordered, row_order, col_order = reorder_matrix(matrix, partition)
    return reordered, row_order, col_order


def format_matrix(matrix: npt.ArrayLike) -> npt.ArrayLike:
    """
    Format the matrix for display, using '*' for non-zero elements.
    """
    formatted = np.full(matrix.shape, " ", dtype=object)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] != 0:
                formatted[i, j] = "*"
    return formatted


def run_balp_example() -> tuple[npt.ArrayLike, npt.ArrayLike, list[int], list[int]]:
    """
    Run the BALP transformation on the example from the image.
    """
    # Create the example matrix from the second image
    # Original matrix with rows 1-5 and columns 1-7
    # The stars in the matrix represent non-zero entries
    # From the example in the second image:
    original = np.zeros((5, 7))

    # Fill in the non-zero entries based on the example
    # Row 1 has entries in columns 1, 3, 4
    original[0, 0] = 1  # (1,1)
    original[0, 2] = 1  # (1,3)
    original[0, 3] = 1  # (1,4)

    # Row 2 has entries in columns 3, 5
    original[1, 2] = 1  # (2,3)
    original[1, 4] = 1  # (2,5)

    # Row 3 has entries in columns 4, 7
    original[2, 3] = 1  # (3,4)
    original[2, 6] = 1  # (3,7)

    # Row 4 has entries in columns 1, 2, 6
    original[3, 0] = 1  # (4,1)
    original[3, 1] = 1  # (4,2)
    original[3, 5] = 1  # (4,6)

    # Row 5 has entry in column 5
    original[4, 4] = 1  # (5,5)

    print("Original Matrix:")
    print(original)

    # Create bipartite graph
    G = create_bipartite_graph(original)

    # Visualize the original bipartite graph
    # visualize_bipartite_graph(G, "Original Bipartite Graph")

    # Partition the graph
    partition = partition_graph(G)

    # Identify blocks and reorder matrix
    reordered, row_order, col_order = identify_blocks(original, partition)

    print("\nReordered Matrix:")
    print(reordered)

    print("\nRow Order: (Original -> New)")
    for i, r in enumerate(row_order):
        print(f"{r+1} -> {i+1}")

    print("\nColumn Order: (Original -> New)")
    for i, c in enumerate(col_order):
        print(f"{c+1} -> {i+1}")

    # Format matrices for display
    formatted_original = format_matrix(original)
    formatted_reordered = format_matrix(reordered)

    print("\nOriginal Matrix (formatted):")
    print_formatted_matrix(formatted_original, list(range(1, 6)), list(range(1, 8)))

    print("\nReordered Matrix (formatted):")
    print_formatted_matrix(
        formatted_reordered,
        [row_order[i] + 1 for i in range(len(row_order))],
        [col_order[i] + 1 for i in range(len(col_order))],
    )

    return original, reordered, row_order, col_order


def print_formatted_matrix(
    matrix: npt.ArrayLike, row_labels: list[int], col_labels: list[int]
) -> None:
    """
    Print a formatted matrix with row and column labels.
    """
    # Print column headers
    print("   ", end="")
    for j in col_labels:
        print(f"{j:2}", end=" ")
    print()

    # Print rows with labels
    for i, row in enumerate(matrix):
        print(f"{row_labels[i]:2} ", end="")
        for cell in row:
            print(f"{cell:2}", end=" ")
        print()


if __name__ == "__main__":
    original, reordered, row_order, col_order = run_balp_example()
