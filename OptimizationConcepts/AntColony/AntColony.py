from geopy.geocoders import Nominatim
import numpy as np
from pylab import *
import random

# city class
class City:
    def __init__(self, x, y, name):
        self.x = x
        self.y = y
        self.name = name

    def distance(self, City):
        if self == City:
            return float('inf')

        return np.sqrt((self.x - City.x) ** 2 + (self.y - City.y) ** 2)

    def getPosition(self):
        return self.x, self.y

    def getName(self):
        return self.name

    def __str__(self):
        return f"{self.name}: ({self.x}, {self.y})"


# Ant class
class Ant:
    def __init__(self, start_city=0):
        self.tour = [start_city]
        self.visited = set([start_city])
        self.length = 0
        self.current_city = start_city

    def visit_city(self, city, distance):
        self.tour.append(city)
        self.visited.add(city)
        self.length += distance
        self.current_city = city

    def reset(self, start_city=None):
        if start_city is not None:
            self.current_city = start_city
        self.tour = [self.current_city]
        self.visited = set([self.current_city])
        self.length = 0


def saveCityCoordinates(inputFile="./India_cities.txt", outputFile="./India_cities_coords.txt"):
    geoLocator = Nominatim(user_agent="ssm_app")

    with open(inputFile) as file, open(outputFile, "w") as out:
        j = 0
        for line in file:
            city = line.strip()
            if city == "":
                break

            thelocation = city + ", India"
            pt = geoLocator.geocode(thelocation, timeout=1000)
            y = round(pt.latitude, 2)
            x = round(pt.longitude, 2)

            print(f"City[{j}] = {city} ({x:.2f}, {y:.2f})")
            out.write(f"{city},{x},{y}\n")
            j += 1

    print(f"✅ Saved coordinates to {outputFile}")


# function to read cities from a txt file
def readCities(coordsFile="./India_cities_coords.txt"):
    cityList = []
    with open(coordsFile) as file:
        for line in file:
            parts = line.strip().split(",")
            if len(parts) != 3:
                continue
            name, x, y = parts[0], float(parts[1]), float(parts[2])
            cityList.append(City(x, y, name))
    return cityList


# Function to plot the route
def Plot(seq, dist):
    P = [city.getPosition() for city in seq]
    cityNames = [city.getName() for city in seq]

    Pt = np.array(P + [P[0]])

    plt.figure(figsize=(8, 6))
    plt.title("Best Route | Total Distance = {:.2f}".format(dist))
    plt.plot(Pt[:, 0], Pt[:, 1], "-o")

    for i, name in enumerate(cityNames):
        plt.annotate(name, (P[i][0], P[i][1]))

    plt.show()


# function to calculate the distance matrix
def calculateDistMatrix(cityList, N):
    distMatrix = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i == j:
                distMatrix[i][j] = float('inf')
            else:
                distMatrix[i][j] = cityList[i].distance(cityList[j])

    return distMatrix


# function to build tour
def buildTour(ant, numCities, pheromoneMatrix, distMatrix, ALPHA, BETA):
    # reset each ant (optional if reusing)
    ant.reset()

    # construct a full tour
    while len(ant.visited) < numCities:
        current_city = ant.current_city

        # list of cities not yet visited
        unvisited = [c for c in range(numCities) if c not in ant.visited and c != current_city]

        # compute probabilities for each unvisited city
        tau = np.array([pheromoneMatrix[current_city][c] ** ALPHA for c in unvisited])
        eta = np.array([1.0 / (distMatrix[current_city][c] + 1e-6) ** BETA for c in unvisited])
        probs = tau * eta
        sum_probs = np.sum(probs)

        if sum_probs == 0:
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / sum_probs

        # roulette wheel selection
        r = random.random()
        cumulative = 0.0
        for city, p in zip(unvisited, probs):
            cumulative += p
            if r <= cumulative:
                ant.visit_city(city, distMatrix[current_city][city])
                break


    # return to starting city
    start_city = ant.tour[0]
    ant.length += distMatrix[ant.current_city][start_city]
    ant.tour.append(start_city)


# function to update the pheromone
def updatePheromone(ants, pheromoneMatrix, Q, RHO):
    # Evaporation
    pheromoneMatrix *= (1 - RHO)

    # Deposit pheromone
    for ant in ants:
        for i in range(len(ant.tour) - 1):
            a, b = ant.tour[i], ant.tour[i + 1]
            pheromoneMatrix[a][b] += Q / ant.length
            pheromoneMatrix[b][a] += Q / ant.length

    return pheromoneMatrix


# function to check the best tour
def checkBest(ants, best_length, best_tour):
    best_ant = min(ants, key=lambda a: a.length)

    if best_length is None or best_ant.length < best_length:
        best_length = best_ant.length
        best_tour = best_ant.tour.copy()

    return best_length, best_tour


# main function for ant colony optimization
def antColonyOptimization(ALPHA, BETA, RHO, Q, numAnts, numIterations, TAUO):
    cityList = readCities()
    N = len(cityList)

    pheromoneMatrix =  np.full((N, N), TAUO)
    distMatrix = calculateDistMatrix(cityList=cityList, N=N)

    start_positions = random.sample(range(N), numAnts)

    ants = [Ant(start_city=start) for start in start_positions]

    best_length = None
    best_tour = []

    for iteration in range(numIterations):
        for ant in ants:
            buildTour(ant, N, pheromoneMatrix, distMatrix, ALPHA, BETA)
            
        pheromoneMatrix = updatePheromone(ants, pheromoneMatrix, Q, RHO)
        best_length, best_tour = checkBest(ants, best_length, best_tour)

        print(f"Best length this iteration ({iteration+1}): {best_length}")
        print()

        # Plotting graph every 25 iterations
        if (iteration + 1) % 25 == 0:
            bestTour = [cityList[i] for i in best_tour]
            Plot(bestTour, best_length)

        # Reseting the ants
        for ant in ants:
            ant.reset(start_city=random.choice(range(N)))

    # final plot
    bestTour = [cityList[i] for i in best_tour]
    Plot(bestTour, best_length)



if __name__ == "__main__":
    antColonyOptimization(ALPHA=1.0, BETA=5.0, RHO=0.9, Q=100, numAnts=40, numIterations=100, TAUO=1.0)