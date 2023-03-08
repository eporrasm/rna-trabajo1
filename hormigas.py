import random
import math
import operator
import matplotlib.pyplot as plt
import pandas as pd

df_ciudades = pd.read_csv("ciudades.csv")
df_tiempos = pd.read_csv("tiempos.csv")
df_peajes = pd.read_csv("peajes.csv")
df_distancias = pd.read_csv("distancias.csv")

pago_hora = 6827
df_costo_vendedor = df_tiempos.applymap(lambda x: x*pago_hora if not isinstance(x, str) else x )

consumo = 8.35/100 #litro/km
precio_gasolina = 2747.31 #$/litro
costo_kilometro = precio_gasolina * consumo # $/km
df_costo_combustible = df_distancias.applymap(lambda x: float(x)*costo_kilometro if not isinstance(x, str) else x )

df_costo_total = (df_costo_combustible.drop("Distancia", axis=1)+
                    df_costo_vendedor.drop("Tiempo viaje", axis=1)+
                    df_peajes.drop("Costo Peajes", axis=1))
df_costo_total = df_costo_total.set_index(pd.Series(['Armenia', 'Barranquilla', 'Bogota D.C.', 
                                                        'Bucaramanga', 'Cartagena', 'Cucuta', 
                                                        'Manizales', 'Medellin', 'Monteria', 
                                                        'Palmira', 'Pereira', 'Pasto', 'Soledad', 
                                                        'Tulua', 'Valledupar']))
k = 1

class Graph(object):
    def __init__(self, cost_matrix: list, rank: int):
        """
        :param cost_matrix:
        :param rank: rank of the cost matrix
        """
        self.matrix = cost_matrix
        self.rank = rank
        # noinspection PyUnusedLocal
        self.pheromone = [[1 / (rank * rank) for j in range(rank)] for i in range(rank)]


class ACO(object):
    def __init__(self, ant_count: int, generations: int, alpha: float, beta: float, rho: float, q: int,
                 strategy: int):
        """
        :param ant_count:
        :param generations:
        :param alpha: relative importance of pheromone
        :param beta: relative importance of heuristic information
        :param rho: pheromone residual coefficient
        :param q: pheromone intensity
        :param strategy: pheromone update strategy. 0 - ant-cycle, 1 - ant-quality, 2 - ant-density
        """
        self.Q = q
        self.rho = rho
        self.beta = beta
        self.alpha = alpha
        self.ant_count = ant_count
        self.generations = generations
        self.update_strategy = strategy

    def _update_pheromone(self, graph: Graph, ants: list):
        for i, row in enumerate(graph.pheromone):
            for j, col in enumerate(row):
                graph.pheromone[i][j] *= self.rho
                for ant in ants:
                    graph.pheromone[i][j] += ant.pheromone_delta[i][j]

    # noinspection PyProtectedMember
    def solve(self, graph: Graph):
        """
        :param graph:
        """
        best_cost = float('inf')
        best_solutions = []
        for gen in range(self.generations):
            # noinspection PyUnusedLocal
            ants = [_Ant(self, graph) for i in range(self.ant_count)]
            for ant in ants:
                for i in range(graph.rank - 1):
                    ant._select_next()
                ant.total_cost += graph.matrix[ant.tabu[-1]][ant.tabu[0]]
                if ant.total_cost < best_cost:
                    best_cost = ant.total_cost
                    best_solutions.append(ant.tabu)
                # update pheromone
                ant._update_pheromone_delta()
            self._update_pheromone(graph, ants)
            # print('generation #{}, best cost: {}, path: {}'.format(gen, best_cost, best_solution))
        return best_solutions, best_cost


class _Ant(object):
    def __init__(self, aco: ACO, graph: Graph):
        self.colony = aco
        self.graph = graph
        self.total_cost = 0.0
        self.tabu = []  # tabu list
        self.pheromone_delta = []  # the local increase of pheromone
        self.allowed = [i for i in range(graph.rank)]  # nodes which are allowed for the next selection
        self.eta = [[0 if i == j else 1 / graph.matrix[i][j] for j in range(graph.rank)] for i in
                    range(graph.rank)]  # heuristic information
        start = random.randint(0, graph.rank - 1)  # start from any node
        self.tabu.append(start)
        self.current = start
        self.allowed.remove(start)

    def _select_next(self):
        denominator = 0
        for i in self.allowed:
            denominator += self.graph.pheromone[self.current][i] ** self.colony.alpha * self.eta[self.current][
                                                                                            i] ** self.colony.beta
        # noinspection PyUnusedLocal
        probabilities = [0 for i in range(self.graph.rank)]  # probabilities for moving to a node in the next step
        for i in range(self.graph.rank):
            try:
                self.allowed.index(i)  # test if allowed list contains i
                probabilities[i] = self.graph.pheromone[self.current][i] ** self.colony.alpha * \
                    self.eta[self.current][i] ** self.colony.beta / denominator
            except ValueError:
                pass  # do nothing
        # select next node by probability roulette
        selected = 0
        rand = random.random()
        for i, probability in enumerate(probabilities):
            rand -= probability
            if rand <= 0:
                selected = i
                break
        self.allowed.remove(selected)
        self.tabu.append(selected)
        self.total_cost += self.graph.matrix[self.current][selected]
        self.current = selected

    # noinspection PyUnusedLocal
    def _update_pheromone_delta(self):
        self.pheromone_delta = [[0 for j in range(self.graph.rank)] for i in range(self.graph.rank)]
        for _ in range(1, len(self.tabu)):
            i = self.tabu[_ - 1]
            j = self.tabu[_]
            if self.colony.update_strategy == 1:  # ant-quality system
                self.pheromone_delta[i][j] = self.colony.Q
            elif self.colony.update_strategy == 2:  # ant-density system
                # noinspection PyTypeChecker
                self.pheromone_delta[i][j] = self.colony.Q / self.graph.matrix[i][j]
            else:  # ant-cycle system
                self.pheromone_delta[i][j] = self.colony.Q / self.total_cost

def distance(city1: dict, city2: dict):
    return math.sqrt((city1['x'] - city2['x']) ** 2 + (city1['y'] - city2['y']) ** 2)

def plot(points, path: list):
    global k
    background = plt.imread('mapa-colombia.png')
    plt.imshow(background, extent=[0, 740, 0, 740])
    x = []
    y = []
    for point in points:
        x.append(point[0])
        y.append(point[1])
    # noinspection PyUnusedLocal
    #y = list(map(operator.sub, [max(y) for i in range(len(points))], y))
    plt.plot(x, y, 'co')

    for _ in range(1, len(path)):
        i = path[_ - 1]
        j = path[_]
        # noinspection PyUnresolvedReferences
        plt.arrow(x[i], y[i], x[j] - x[i], y[j] - y[i], color='r', length_includes_head=True)

    # noinspection PyTypeChecker
    plt.xlim(0, 740)
    # noinspection PyTypeChecker
    plt.ylim(0, 740)
    plt.savefig("imagenes_hormigas/mapa{}.png".format(k))
    k += 1
    plt.show()



def main():
    nombres = list(df_ciudades.Municipio)
    X = [266, 297, 330, 367, 274, 392, 270, 267, 261, 240, 262, 196, 296, 247, 362]
    Y = [378, 648, 389, 485, 620, 520, 406, 457, 553, 343, 394, 248, 361, 364, 626]
    cities = []
    points = []
    i = 0
    for x, y in zip(X, Y):
        cities.append(dict(index=i, x=x, y=y))
        points.append((int(x), int(y)))
        i += 1
    cost_matrix = df_costo_total.values.tolist()
    rank = len(cities)
    for i in range(rank):
        for j in range(rank):
            if i != j and cost_matrix[i][j] == 0:
                cost_matrix[i][j] = 20000
    aco = ACO(100, 200, 1.0, 1.0, 0.1, 10, 2)
    graph = Graph(cost_matrix, rank)
    paths, cost = aco.solve(graph)
    print('costo: ${}, recorrido: {}'.format(round(cost), [nombres[i] for i in paths[-1]]))
    print(len(paths))
    for path in paths:
        plot(points, path)

main()