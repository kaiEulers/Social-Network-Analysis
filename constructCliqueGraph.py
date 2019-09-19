"""
@author: kaisoon
"""
import pickle
import time
import networkx as nx

def constructCG(G, CLIQUES):
    """
    :param G:
    :param CLIQUES:
    :return:
    """
    # ================================================================================
    # ----- FOR DEBUGGING
    DATE = '2017-12'
    PATH = f"results/{DATE}/"

    # PARAMETERS
    # G = nx.read_gpickle(f"{PATH}ssm_{DATE}_weightedGraph.gpickle")
    # with open(f"{PATH}ssm_{DATE}_cliques.pickle", "rb") as file:
    #     CLIQUES = pickle.load(file)
    # ================================================================================
    # ----- Construct clique graphs
    startTime = time.time()
    graphNodes = dict(G.nodes)
    CGs = {}
    for k, clq in CLIQUES.items():
        CG = nx.Graph()
        CG.clear()
        # ----- Add nodes
        print(f"Adding nodes for cliqueGraph{k}...")
        for person in clq:
            CG.add_node(
                person,
                party=graphNodes[person]['party'],
                gender=graphNodes[person]['gender'],
                metro=graphNodes[person]['metro'],
                data=graphNodes[person]['data'],
                centrality=graphNodes[person]['centrality'],
                cliques=graphNodes[person]['cliques'],
            )
        print(f"All nodes of cliqueGraphs successfully added for cliqueGraph{k}")

        # ----- Add edges
        print(f"Adding edges for cliqueGraph{k}...")
        for i, p_i in enumerate(clq):
            for j, p_j in enumerate(clq, start=i+1):
                # Check that p_i and p_j are not the same people
                if p_i != p_j:
                    # Get edge data
                    edgeData = G.get_edge_data(p_i, p_j)
                    CG.add_edge(p_i, p_j, weight=edgeData['weight'], agreedSpeeches=edgeData['agreedSpeeches'])

        CGs[k] = CG
        print(f"All edges of cliqueGraphs successfully added for cliqueGraph{k}\n")


    print(f"Clique graph construction complete!")
    print(f"Construction took {round(time.time()-startTime, 2)}s")

    # ================================================================================
    # ----- FOR DEBUGGING
    # Save clique graph
    nx.write_gpickle(CGs, f"{PATH}ssm_{DATE}_cliqueGraphs.gpickle")
    # ================================================================================

    return CGs
