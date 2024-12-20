from heapq import heappush, heappop
import random
import matplotlib.pyplot as plt
from timeit import timeit
import math
import copy
import itertools

def SP_naive (graph, s):
    '''
    Shortest path algorithm with naive implementation.
    graph: adjacency list of the graph
    s: source vertex
    return: a dictionary of shortest distances from s to all other vertices
    '''
    frontier = [s]
    dist = {s:0}

    while len(frontier) > 0:

        x = min(frontier, key = lambda k: dist[k])
        frontier.remove(x)

        for y, dxy in graph[x].items():
            dy = dist[x] + dxy

            if y not in dist:
                frontier.append(y)
                dist[y] = dy

            elif dist[y] > dy:
                dist[y] = dy

    return dist

# Shortest path algorithm with binary heap

def SP_heap (graph, s):
    '''
    Shortest path algorithm with binary heap.
    graph: adjacency list of the graph
    s: source vertex
    return: a dictionary of shortest distances from s to all other vertices
    '''
    frontier = []
    heappush(frontier, (0, s))
    done = set()
    dist = {s: 0}

    while len(frontier) > 0:

        dx, x = heappop(frontier)
        if x in done:
            continue

        done.add(x)

        for y, dxy in graph[x].items():
            dy = dx + dxy

            if y not in dist or dist[y] > dy:
                heappush(frontier,(dy, y))
                dist[y] = dy

    return dist

####################### Question 1 #####################
def q1():
    return '''
    O(V^2) in both cases
    '''

####################### Question 2 #####################
def q2():
    return '''
    sparse graph : O(V log(V))
    dense graph : O(V^2 log(V))
    '''

def random_sparse_graph (n, step):
    '''
    Generate a random sparse graph with n vertices.
    n: number of vertices
    step: maximum distance between two adjacent vertices
    return: adjacency list of the graph
    '''
    graph = {f'{i}_{j}': {} for i in range(1, n+1) for j in range(1, n+1)}

    for i in range(1, n+1):
        for j in range(1, n):
            d = random.randint(step+1, 2*step)
            graph[f'{i}_{j}'][f'{i}_{j+1}'] = d
            graph[f'{i}_{j+1}'][f'{i}_{j}'] = d

    for i in range(1, n):
        for j in range(1, n+1):
            d = random.randint(step+1, 2*step)
            graph[f'{i}_{j}'][f'{i+1}_{j}'] = d
            graph[f'{i+1}_{j}'][f'{i}_{j}'] = d

    return graph

def random_dense_graph (n, d_max):
    '''
    Generate a random dense graph with n vertices.
    n: number of vertices
    d_max: maximum distance between two adjacent vertices
    return: adjacency list of the graph
    '''
    graph = {f'{i}':{} for i in range(n)}

    for n1 in graph:
        for n2 in graph:
            if n2!= n1 and n2 not in graph[n1]:
                d = random.randint(1, d_max)
                graph[n1][n2] = d
                graph[n2][n1] = d

    return graph

def benchmark():
    Time_heap_sparse = []
    Time_naive_sparse = []
    Time_heap_dense = []
    Time_naive_dense = []

    n_list = []

    for N in range(10, 30):
        print(f'N={N} is being processed...')
        n = N*N
        n_list.append(n)

        # on compare sur des cartes non-denses: grille de côté N contient N*N villes
        graph_sparse = random_sparse_graph(n = N, step = 100)

        # on calcule une moyenne sur N lancements en tirant aléatoirement une ville de départ à chaque fois
        Time_naive_sparse.append(timeit(lambda: SP_naive(graph_sparse, random.choice(list(graph_sparse))), number=N) / N)
        Time_heap_sparse.append(timeit(lambda: SP_heap(graph_sparse, random.choice(list(graph_sparse))), number=N) / N)

        # on compare sur des cartes denses
        graph_dense = random_dense_graph(n = N*N, d_max = 10000)

        Time_naive_dense.append(timeit(lambda: SP_naive(graph_dense, random.choice(list(graph_dense))), number=N) / N)
        Time_heap_dense.append(timeit(lambda: SP_heap(graph_dense, random.choice(list(graph_dense))), number=N) / N)

    plt.xlabel('N')
    plt.ylabel('T')
    plt.plot(n_list, Time_naive_sparse, 'r^', label="naive sparse")
    plt.plot(n_list, Time_heap_sparse, 'b^', label="heap sparse")
    plt.plot(n_list, Time_naive_dense, 'r*', label="naive dense")
    plt.plot(n_list, Time_heap_dense, 'b*', label="heap dense")

    plt.legend()
    plt.show()
    
# From your answer to the precedent question, what structure has the 
# better complexity for dense graphs in theory? What you observe from 
# the result of the benchmark? Can you explain this?

####################### Question 3 #####################
def q3():
    print("Benchmarking...")
    benchmark()
    return '''
    In theory, the naive algorithm should be faster for dense graphs.
    In the benchmark it is not the case, probably because dense graphs do not exactly verify E=V^2.
    '''

# For sparse graphs, does the running result of the benchmark for the naive 
# structure (list) correspond to the theoretical time complexity? 
# Can you explain this?

def q4():
    return '''
    For sparse graphs, the naive algorithm should in theory be O(V^2) but the benchmark is almost constant witch is better.
    However, the naive algorithm was expected to run slower than the heap one for sparse graphs, witch is the case.
    '''

# What is your general conclusion?
def q5():
    return '''
    In a nutshell, the heap algorithm is allways better than the naive algorithm.
    '''

####################### Question 6 #####################

def add_source(graph, src):
    '''
    Add a source vertex to the graph.
    graph: adjacency list of the graph
    src: source vertex
    return: adjacency list of the graph with the source vertex
    '''
    if src in graph:
        return "error: the source vertex is already in the graph"

    graph_src=graph.copy()

    graph_src[src]={}
    for v in graph.keys():
        graph_src[src][v]=0

    return graph_src

sim_graph = {"A":{"B":4,"C":2,"D":3},
             "B":{"A":6,"C":-5},
             "C":{"D":1},
             "D":{}}

src="source"

sim_graph_src=add_source(sim_graph, src)

assert sim_graph_src == {"A":{"B":4,"C":2,"D":3},
                         "B":{"A":6,"C":-5},
                         "C":{"D":1},
                         "D":{},
                         "source":{"A":0, "B":0, "C":0, "D":0}}


####################### Question 7 #####################

def bellman_ford(graph, src):
    '''
    Bellman-Ford algorithm to find the shortest path from a source vertex to all other vertices.
    graph: adjacency list of the graph
    src: source vertex
    return: a dictionary of shortest distances from the source vertex to all other vertices
    '''
    n=len(graph)

    dist={}

    #initialize dist
    ############TODO : complete code#############
    for i in graph:
        dist[i]=math.inf
    dist[src]=0

    # calculate optimal distance
    V = len(graph)
    for _ in range(V - 1):
        for u in graph:
            for v, weight in graph[u].items():
                if dist[u] + weight < dist[v]:
                    dist[v] = dist[u] + weight

    # detect negative cycle: return None if there is any
    for u in graph:
        for v, weight in graph[u].items():
            if dist[u] + weight < dist[v]:
                # Negative weight cycle detected
                return None

    return dist

opt_distance=bellman_ford(sim_graph_src, src)

assert opt_distance=={'A': 0, 'B': 0, 'C': -5, 'D': -4, 'source': 0}

neg_cycle_graph = {"A":{"B":-5,"C":2,"D":3},
             "B":{"A":3,"C":4},
             "C":{"D":1},
             "D":{}}


assert bellman_ford(neg_cycle_graph, "A")==None

####################### Question 8 #####################

def rewrite_weights(graph, dist):
    '''
    Rewrite the weights of the graph to make them nonnegative.
    graph: adjacency list of the graph
    dist: a dictionary of shortest distances from the source vertex to all other vertices
    return: adjacency list of the graph with nonnegative weights
    '''
    # use deepcopy
    altered_graph = copy.deepcopy(graph)

    # Recalculate the new nonnegative weights
    for i in altered_graph.keys():
        for j in altered_graph[i].keys():
            altered_graph[i][j]=altered_graph[i][j]+dist[i]-dist[j]

    return altered_graph

opt_distance={'A': 0, 'B': 0, 'C': -5, 'D': -4, 'source': 0}

nonneg_graph=rewrite_weights(sim_graph, opt_distance)

assert nonneg_graph=={'A': {'B': 4, 'C': 7, 'D': 7}, 'B': {'A': 6, 'C': 0}, 'C': {'D': 0}, 'D': {}}

####################### Question 9 #####################
def q9():
    return '''
    Because dist[i]-dist[j] is greater or equal to each weight thanks to the Bellman Ford function.
    '''

####################### Question 10 #####################
def q10():
    return '''
    Because the dictionnary dist was already calculated for each vertex.
    '''

####################### Question 11 #####################

def all_distances(graph):
    '''
    Compute all shortest distances between all pairs of vertices in the graph.
    graph: adjacency list of the graph
    return: a dictionary of all shortest distances between all pairs of vertices
    '''
    d = {(u,v):None for u in graph for v in graph}

    ############### TODO : complete code ##################
    for source in graph:
        shortest_paths = SP_heap(graph, source)
        
        for target, distance in shortest_paths.items():
            d[(source, target)] = distance

    return d

nonneg_graph={'A': {'B': 4, 'C': 7, 'D': 7}, 'B': {'A': 6, 'C': 0}, 'C': {'D': 0}, 'D': {}}


assert all_distances(nonneg_graph)=={('A', 'A'): 0, ('A', 'B'): 4, ('A', 'C'): 4, ('A', 'D'): 4, 
                                     ('B', 'A'): 6, ('B', 'B'): 0, ('B', 'C'): 0, ('B', 'D'): 0, 
                                     ('C', 'A'): None, ('C', 'B'): None, ('C', 'C'): 0, ('C', 'D'): 0, 
                                     ('D', 'A'): None, ('D', 'B'): None, ('D', 'C'): None, ('D', 'D'): 0}


####################### Question 12 #####################

def BF_SP_all_pairs(graph, src="source"):

    ############### TODO : complete code ##################
    d = {(u,v):None for u in graph for v in graph}


    arbitrary_source = list(graph.keys())[0]
    h = bellman_ford(graph, arbitrary_source)

    rewritten_graph = rewrite_weights(graph, h)

    distances = all_distances(rewritten_graph)


    for (u, v), distance in distances.items():
        if distance is not None:
            d[(u, v)] = distance + h[v] - h[u]

    return d

assert BF_SP_all_pairs(sim_graph)=={('A', 'A'): 0, ('A', 'B'): 4, ('A', 'C'): -1, ('A', 'D'): 0, 
                                    ('B', 'A'): 6, ('B', 'B'): 0, ('B', 'C'): -5, ('B', 'D'): -4, 
                                    ('C', 'A'): None, ('C', 'B'): None, ('C', 'C'): 0, ('C', 'D'): 1, 
                                    ('D', 'A'): None, ('D', 'B'): None, ('D', 'C'): None, ('D', 'D'): 0}


####################### Question 13 #####################

def q13():
    return '''
    O(V^2 log(V))
    '''

####################### Question 14 #####################

def closest_oven(house, oven_houses, distance_dict):
    '''
    Find the closest oven house to a given house.
    house: a house
    oven_houses: a list of oven houses
    distance_dict: a dictionary of all shortest distances between all pairs of vertices
    return: a tuple of the distance to the closest oven house and the closest oven house
    '''
    ############### TODO : complete code ##################


    if not oven_houses:
        return (math.inf, None)

    min_distance = math.inf
    nearest_house = None

    for oven_house in oven_houses:
        distance = distance_dict.get((house, oven_house), math.inf)
        
        if distance < min_distance:
            min_distance = distance
            nearest_house = oven_house
    return (min_distance, nearest_house)


toy_village = {'A': {'B': -3, 'E': 20, 'F': 30},
           'B': {'A': 6, 'C': 9, 'F': 39},
           'C': {'B': 9, 'D': 8},
           'D': {'C': 8, 'F': 50},
           'E': {'A': -10, 'F': 6},
           'F': {'A': -20, 'B': -25, 'D': -15, 'E': 6} }

toy_distance_dict = BF_SP_all_pairs(toy_village)

assert closest_oven('A', ['B','E','D'], toy_distance_dict) == (-3, 'B')
assert closest_oven('B', ['B','E','D'], toy_distance_dict) == (0, 'B')
assert closest_oven('C', ['B','E','D'], toy_distance_dict) == (8, 'D')
assert closest_oven('D', ['B','E','D'], toy_distance_dict) == (0, 'D')
assert closest_oven('E', ['B','E','D'], toy_distance_dict) == (-19, 'B')
assert closest_oven('F', ['B','E','D'], toy_distance_dict) == (-25, 'B')

####################### Question 15 #####################

def kcentre_value(village, oven_houses, distance_dict):
    '''
    Compute the maximum distance between a house and the closest oven house.
    village: adjacency list of the village
    oven_houses: a list of oven houses
    distance_dict: a dictionary of all shortest distances between all pairs of vertices
    return: the maximum distance between a house and the closest oven house
    '''

    ############### TODO : complete code ##################
    max_distance = -math.inf

    for house in village:
        min_distance, _ = closest_oven(house, oven_houses, distance_dict)
        
        if min_distance > max_distance:
            max_distance = min_distance

    return max_distance

assert kcentre_value(toy_village, ['B','E','D'], toy_distance_dict) == 8
####################### Question 16 #####################

def read_map(filename):
    '''
    Read a map from a file.
    filename: name of the file
    return: adjacency list of the map
    '''
    f = open(file=filename, mode='r', encoding='utf-8')

    map = {}
    while True:  # reading list of cities from file
        ligne = f.readline().rstrip()
        if (ligne == '--'):
            break
        info = ligne.split(':')
        map[info[0]] = {}

    while True:  # reading list of distances from file
        ligne = f.readline().rstrip()
        if (ligne == ''):
            break
        info = ligne.split(':')
        map[info[0]][info[1]] = int(info[2])

    return map

def brute_force(map, candidates, k, distance_dict) :
    best_combi = []
    best_dist = math.inf
    for combi in itertools.combinations(candidates, k):
        dist= kcentre_value(map, list(combi), distance_dict)
        if  dist<best_dist:
            best_combi= list(combi)
            best_dist= dist
    return  best_dist, set(best_combi)

def BF_benchmark():

    village = read_map('village.map')

    village_distance_dict = BF_SP_all_pairs(village)

    # assert brute_force(village, list(village), 3, village_distance_dict) == (0, {'C', 'D', 'B'})
    # assert brute_force(village, list(village), 2, village_distance_dict) == (8, {'C', 'A'})

    Time_brute_force = []

    k_list = []

    for k in range(1, 20):
        print(f'k={k} is being processed...')
        Time_brute_force.append(timeit(lambda: brute_force(village, list(village), k, village_distance_dict), number=1))
        k_list.append(k)

    #print N_list, time_list
    plt.xlabel('k')
    plt.ylabel('T')
    plt.plot(k_list, Time_brute_force, 'r^')
    plt.xticks(k_list)
    plt.show()
    
def q16():
    print("Benchmarking...")
    BF_benchmark()
    return'''
    The algorithm gets very slow for k around 10, but is efficient pour small or large values of k.
    This comes from the combinatory (20 choose k) is maximal for k=10.
    Because the bruteforce algorithm passes throught every case.
    '''

####################### Question 17 #####################

def greedy_algorithm(map, candidates, k, distance_dict):
    '''
    Greedy algorithm to find the k-centre of a village.
    map: adjacency list of the village
    candidates: list of houses in the village
    k: number of oven houses
    distance_dict: a dictionary of all shortest distances between all pairs of vertices
    return: a tuple of the maximum distance between a house and the closest oven house and the set of oven houses
    '''
    ############### TODO : complete code ##################


    selected_houses = []
    
    total_cost = 0
    
    for _ in range(k):
        best_cost = float('inf')
        best_house = None
        
        for house in candidates:
            if house not in selected_houses:
                selected_houses.append(house)
                current_cost= kcentre_value(map, list(selected_houses+[house]), distance_dict)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_house = house
                selected_houses.remove(house)
        
        selected_houses.append(best_house)
        total_cost = best_cost

    return total_cost, selected_houses

village = read_map('village.map')
village_distance_dict = BF_SP_all_pairs(village)
force_d, force_h = brute_force(village, list(village), 5, village_distance_dict)
greed_d, greed_h = greedy_algorithm(village, list(village), 5, village_distance_dict)  
assert greed_d >= force_d
assert brute_force(village, list(village), 1, village_distance_dict) == (14, {'CL5'})
assert brute_force(village, list(village), 5, village_distance_dict) == (2, {'CL4', 'CL7', 'CL6', 'CL5', 'CL2'})
print(greed_d, force_d)

        
####################### Question 18 #####################

def random_algorithm(map, candidates, k, distance_dict, trials=100) :
    '''
    Random algorithm to find the k-centre of a village.
    map: adjacency list of the village
    candidates: list of houses in the village
    k: number of oven houses
    distance_dict: a dictionary of all shortest distances between all pairs of vertices
    trials: number of trials
    return: a tuple of the maximum distance between a house and the closest oven house and the set of oven houses
    '''
    ############### TODO : complete code ##################
    total_cost = float('inf')
    selected_houses=[]
    for _ in range(trials):
        houses_sample=random.sample(candidates, k)
        current_cost= kcentre_value(map, list(houses_sample), distance_dict)
        if current_cost<total_cost:
            total_cost=current_cost
            selected_houses=houses_sample
    return total_cost, selected_houses




####################### Question 19 #####################

def BF_G_R_benchmark(max_k, random_trials = 100):
    '''
    Benchmark the brute force, greedy and random algorithms.
    max_k: maximum number of oven houses
    random_trials: number of trials for the random algorithm
    '''
    village = read_map('village.map')

    village_distance_dict = BF_SP_all_pairs(village)

    # compare on lite
    time_bruteforce_lite = []
    time_random_lite = []
    time_greedy_lite = []
 

    d_bruteforce_lite = []
    d_random_lite = []
    d_greedy_lite = []
 

    k_list = []


    for k in range(2, max_k):
        print(f'k={k} is being processed...')
        k_list.append(k)


        time_bruteforce_lite.append(timeit(lambda: d_bruteforce_lite.append(brute_force(village, list(village), k, village_distance_dict)[0]), number=1))

        time_greedy_lite.append(timeit(lambda: d_greedy_lite.append(greedy_algorithm(village, list(village), k, village_distance_dict)[0]), number=1))
        time_random_lite.append(timeit(lambda: d_random_lite.append(random_algorithm(village, list(village), k, village_distance_dict, random_trials)[0]), number=1))


    plt.subplot(2, 1, 1)
    plt.xlabel('k')
    plt.ylabel('T')
    plt.xticks(k_list)
    plt.plot(k_list, time_bruteforce_lite, 'r*')
    plt.plot(k_list, time_greedy_lite, 'b*')
    plt.plot(k_list, time_random_lite, 'y*')

    plt.subplot(2, 1, 2)
    plt.xlabel('k')
    plt.ylabel('d')
    plt.xticks(k_list)
    plt.plot(k_list, d_bruteforce_lite, 'r-')
    plt.plot(k_list, d_greedy_lite, 'b-')
    plt.plot(k_list, d_random_lite, 'y-')
    plt.show()
    
def q19():
    print("Benchmarking...")
    BF_G_R_benchmark(20)
    return '''
    For the time complexity, the greedy algorithm and the random algorithm are both pretty efficient.
    But the bruteforce algorithm is not.

    In term of distances, the bruteforce algorithm allways gives the best awnser (lowest distance).
    For small values of k, the greedy algorithm is better than the random algorithm but for large values of k it is the opposite.
    '''
q19()