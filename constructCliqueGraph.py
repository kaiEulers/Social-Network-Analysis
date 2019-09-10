"""
@author: kaisoon
"""
def constructCG(G, CLIQUES):
    import pickle
    import time
    import networkx as nx
    # =====================================================================================
    # ----- FOR DEBUGGING
    DATE = '2017-12'
    PATH = f"results/{DATE}/"

    # PARAMETERS
    # G = nx.read_gpickle(f"{PATH}ssm_{DATE}_weightedGraph.gpickle")
    # with open(f"{PATH}ssm_{DATE}_cliques.pickle", "rb") as file:
    #     CLIQUES = pickle.load(file)


    # =====================================================================================
    #  Construct clique graphs
    startTime = time.time()
    graphNodes = dict(G.nodes)
    CGs = {}
    CG = nx.Graph()
    for k,clique in CLIQUES.items():
        CG.clear()
        # ----- Add nodes
        print(f"Adding nodes for cliqueGraph{k}...")
        for person in clique:
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
        for i in range(len(clique)):
            p1 = clique[i]
            for j in range(i+1, len(clique)):
                p2 = clique[j]
                # Check that p1 and p2 are not the same people
                if p1 != p2:
                    # Get edge data
                    edgeData = G.get_edge_data(p1, p2)
                    CG.add_edge(p1, p2, weight=edgeData['weight'], agreedSpeeches=edgeData['agreedSpeeches'])

        CGs[k] = CG
        print(f"All edges of cliqueGraphs successfully added for cliqueGraph{k}\n")


    print(f"Clique graph construction complete!")
    print(f"{len(CGs)} were cliques found")
    print(f"Construction took {round(time.time()-startTime, 2)}s")


    # =====================================================================================
    # Save clique graph
    nx.write_gpickle(CGs, f"{PATH}ssm_{DATE}_cliqueGraphs.gpickle")

    return CGs
