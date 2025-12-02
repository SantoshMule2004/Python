from geopy.geocoders import Nominatim
from numpy import *
from math import *
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
    def __init__(self, startCity=0):
        self.tour = [startCity]
        self.visited = set([startCity])
        self.length = 0
        self.currentCity = startCity

    def visitCity(self, city, distance):
        self.tour.append(city)
        self.visited.add(city)
        self.length += distance
        self.currentCity = city

    def reset(self, startCity=None):
        if startCity is not None:
            self.currentCity = startCity
        self.tour = [self.currentCity]
        self.visited = set([self.currentCity])
        self.length = 0


# Function to read city names
def readCities():
    cityList = []
    geoLocator = Nominatim(user_agent="ssm_app")
    j = 0

    with open("./India_cities.txt") as file:
        for line in file:
            city = line.rstrip('\n')
            if(city == ""):
                break

            thelocation = city + ", India"
            pt = geoLocator.geocode(thelocation, timeout=10000)
            y = round(pt.latitude, 2)
            x = round(pt.longitude, 2)
            print ("City[%2d] = %s (%5.2f, %5.2f)" %(j, city, x, y))
            cityList.append(City(x, y, city))
            j += 1
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
        currentCity = ant.currentCity

        # list of cities not yet visited
        unvisited = [c for c in range(numCities) if c not in ant.visited and c != currentCity]

        # compute probabilities for each unvisited city
        tau = np.array([pheromoneMatrix[currentCity][c] ** ALPHA for c in unvisited])
        eta = np.array([1.0 / (distMatrix[currentCity][c] + 1e-6) ** BETA for c in unvisited])
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
                ant.visitCity(city, distMatrix[currentCity][city])
                break

    # return to starting city
    start_city = ant.tour[0]
    ant.length += distMatrix[ant.currentCity][start_city]
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
def checkBest(ants, bestLength, bestTour):
    bestAnt = min(ants, key=lambda a: a.length)

    if bestLength is None or bestAnt.length < bestLength:
        bestLength = bestAnt.length
        bestTour = bestAnt.tour.copy()

    return bestLength, bestTour


# main function for ant colony optimization
def antColonyOptimization(ALPHA, BETA, RHO, Q, numAnts, numIterations, TAUO):
    cityList = readCities()
    N = len(cityList)

    pheromoneMatrix =  np.full((N, N), TAUO)
    distMatrix = calculateDistMatrix(cityList=cityList, N=N)

    startPositions = random.sample(range(N), numAnts)

    ants = [Ant(startCity=start) for start in startPositions]

    bestLength = None
    bestTour = []

    maxNoImprovement = 25   # convergence threshold
    noImprovementCount = 0

    for iteration in range(numIterations):
        for ant in ants:
            buildTour(ant, N, pheromoneMatrix, distMatrix, ALPHA, BETA)
            
        pheromoneMatrix = updatePheromone(ants, pheromoneMatrix, Q, RHO)
        previousBest = bestLength
        bestLength, bestTour = checkBest(ants, bestLength, bestTour)

        # Check for improvement
        if previousBest is None or bestLength < previousBest:
            noImprovementCount = 0
        else:
            noImprovementCount += 1

        print(f"Best length this iteration ({iteration+1}): {bestLength}")
        print()

        # Plotting graph every 25 iterations
        if (iteration + 1) % 25 == 0:
            currBestTour = [cityList[i] for i in bestTour]
            Plot(currBestTour, bestLength)

        # Early stopping if converged
        if noImprovementCount >= maxNoImprovement:
            print(f"\nConverged after {iteration+1} iterations (no improvement for {maxNoImprovement} rounds).")
            break

        # Reseting the ants
        for ant in ants:
            ant.reset(startCity=random.choice(range(N)))

    # final plot
    bestTour = [cityList[i] for i in bestTour]
    Plot(bestTour, bestLength)


if __name__ == "__main__":
    antColonyOptimization(ALPHA=1.0, BETA=5.0, RHO=0.9, Q=100, numAnts=40, numIterations=100, TAUO=1.0)