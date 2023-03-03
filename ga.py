import pandas as pd
import random
import pygame
import sys
import math
import time
import os
#os.environ['SDL_VIDEODRIVER']='dummy'
pygame.display.set_mode((640,480))
pygame.init()

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


totalNum = 15 #Número de municipios a visitar
popNum = 5000 #Población del algoritmo
font = pygame.font.Font("freesansbold.ttf", 15)
WIDTH = 740
HEIGHT = 740
PERCENTAGE = 0.5 #Porcentage de la población actual que pasa a la siguiente generación

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Problema del Vendedor Viajero")
background = pygame.image.load("mapa-colombia.png")

class City:
    def __init__(self, nombre, x, y, i):
        self.nombre = nombre
        self.x = x
        self.y = y
        self.num = i
        self.text = font.render(str(self.num), False, (255, 255, 255))

    def display(self):
        pygame.draw.circle(screen, (255, 0, 0), (self.x, self.y), radius = 5)

#Inicializamos las ciudades
nombres = list(df_ciudades.Municipio)

X = [264, 299, 329, 368, 278, 393, 273, 274, 260, 244, 268, 198, 292, 251, 365]
Y = [359, 97, 354, 258, 123, 217, 337, 275, 186, 397, 348, 494, 110, 377, 119]
cities = [City(nombres[i], X[i], Y[i], i) for i in range(totalNum)] 

class Route:
    def __init__(self):
        self.distance = 0
        self.cityPath = random.sample(list(range(totalNum)), totalNum)

    def display(self):
        for i, cityNum in enumerate(self.cityPath):
            pygame.draw.line(screen, (0, 0, 255), (cities[self.cityPath[i]].x, cities[self.cityPath[i]].y), \
                            (cities[self.cityPath[i-1]].x, cities[self.cityPath[i-1]].y), width=2)
    #Aquí tenemos que incorporar la matriz de costos calculada
    def calcDistance(self):
        distance = 0
        for i, cityNum in enumerate(self.cityPath):
            if i != 0:
                distance += df_costo_total[cities[self.cityPath[i-1]].nombre][cities[self.cityPath[i]].nombre]
        self.distance = distance
        return distance

population = [Route() for i in range(popNum)]

#Función que ordena la población con base en la función de ajuste, o sea, el costo de la ruta
def sortPop():
    global population
    population.sort(key = lambda x: x.distance, reverse = False)
    return

#Función que toma el top PERCENTAGE de la población para una generación particular y produce una 
#nueva población reemplazando los miembros no esenciales con nuevos.return
def crossover():
    global population
    updatedPop = []
    updatedPop.extend(population[: int(popNum*PERCENTAGE)])

    for i in range(popNum - len(updatedPop)):
        index1 = random.randint(0, len(updatedPop) - 1)
        index2 = random.randint(0, len(updatedPop) - 1)
        while index1 == index2:
            index2 = random.randint(0, len(updatedPop) - 1)
        parent1 = updatedPop[index1]
        parent2 = updatedPop[index2]
        p = random.randint(0, totalNum - 1)
        child = Route()
        child.cityPath = parent1.cityPath[:p]
        notInChild = [x for x in parent2.cityPath if not x in child.cityPath]
        child.cityPath.extend(notInChild)
        updatedPop.append(child)
    population = updatedPop
    return

def main():
    global population
    running = True
    counter = 0

    best = random.choice(population)

    minDistance = best.calcDistance()
    '''
    Print the coordinates of the randomly generated points
    
    for city in cities:
            print(city.x, city.y)
    '''
    clock = pygame.time.Clock()
    while True:
        screen.fill((0, 0, 0))
        screen.blit(background, (0, 0))
        best.display()
        if counter >= popNum - 1:
            break
        #print(counter)
        clock.tick(60)
        pygame.display.update()
        screen.fill((0, 0, 0))
        for city in cities:
            city.display()
            screen.blit(city.text, (city.x - 20, city.y - 20))
        for element in population:
            element.calcDistance()

        sortPop()
        crossover()
        
        for element in population:
            if element.distance < minDistance:
                minDistance = element.calcDistance()
                best = element
            elif element.distance == minDistance:
                counter += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()
    print("El costo mínimo es : ${}".format(round(minDistance)))
    print("Un camino viable : {}".format([cities[i].nombre for  i in best.cityPath]))
    best.display()
    pygame.display.update()
    time.sleep(5)

if __name__ == "__main__":
    main()