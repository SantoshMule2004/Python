from geopy.geocoders import Nominatim
from numpy import *
from math import *
from pylab import *
import random


class City:
    def __init__(self, x, y, name):
        self.x = x
        self.y = y
        self.name = name

    def distance(self, City):
        if self == City:
            return 0.0

        return np.sqrt((self.x - City.x) ** 2 + (self.y - City.y) ** 2)
    
    def getPosition(self):
        return self.x, self.y

    def getName(self):
        return self.name
    
    def __str__(self):
        print(f"X: {self.x}")
        print(f"Y: {self.y}")
        print(f"city name: {self.name}")
    

class Fitness:
    def __init__(self, route):
        self.route = route
        self.totalDistance = self.pathLength()
        self.fitness = self.pathFitness()

    def pathLength(self):
        dist = 0.0
        N = len(self.route)
        for i in range(N - 1):
            dist += self.route[i].distance(self.route[i+1])

        dist += self.route[i].distance(self.route[0])
        return dist
    
    def pathFitness(self):
        return 1/self.totalDistance


# Function to read city names  
def readCities():
    cityList = []
    geoLocator = Nominatim(user_agent="ssm_app")
    j = 0

    with open("./India_cities_GA.txt") as file:
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

    plt.figure(figsize=(8,6))
    plt.title("Best Route | Total Distance = {:.2f}".format(dist))
    plt.plot(Pt[:, 0], Pt[:, 1], '-o')

    for i, name in enumerate(cityNames):
        plt.annotate(name, (P[i][0], P[i][1]))

    plt.show()


# creates random path for a cityList
def createPath(cityList):
    N = len(cityList)
    return random.sample(cityList, N)


# creates initial population 
def initialPolpulation(popSize, cityList):
    return [createPath(cityList) for _ in range(popSize)]


# function to sort population
def sortPopulationByFitness(population):
    # Evaluate fitness for all
    fitnesses = [Fitness(route).fitness for route in population]
    
    # Sort population by fitness descending
    popFitness = list(zip(population, fitnesses))
    popFitness.sort(key=lambda x: x[1], reverse=True)
    sortedPop = [pf[0] for pf in popFitness]

    return sortedPop


# function to select fittest individuals 
def selection(sortedPopulation, numPairs):
    return sortedPopulation[:2*numPairs]


# crossover
def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    
    childP1 = parent1[start:end+1]
    childP2 = [city for city in parent2 if city not in childP1]
    
    child = childP2[:start] + childP1 + childP2[start:]
    return child


# breeds the population
def breedNewChildren(elitePopulation):
    offspring = []
    for i in range(0, len(elitePopulation), 2):
        parent1 = elitePopulation[i]
        parent2 = elitePopulation[i+1]

        child1 = crossover(parent1, parent2)
        child2 = crossover(parent2, parent1)

        offspring.append(child1)
        offspring.append(child2)
    
    return offspring

 
def mutate(route, mutationRate):
    newRoute = route.copy()
    for i in range(len(newRoute)):
        if random.random() < mutationRate:
            j = random.randint(0, len(newRoute)-1)
            newRoute[i], newRoute[j] = newRoute[j], newRoute[i]
    return newRoute


# Function to mutate population
def mutatepopulation(population, mutationRate):
    mutatedOffspring = []
    for route in population:
        c = mutate(route, mutationRate)
        mutatedOffspring.append(c)
    return mutatedOffspring


# Function to maintain the population 
def trimPopulation(population, originalSize):
    sortedPop = sortPopulationByFitness(population)
    return sortedPop[:originalSize]


def nextGeneration(population, numPairs, popSize, mutationRate):
    sortedPopulaion = sortPopulationByFitness(population)
    parents = selection(sortedPopulaion, numPairs)
    offsprings = breedNewChildren(parents)
    mutatedOffspring = mutatepopulation(offsprings, mutationRate)
    newPopulation = population + mutatedOffspring
    finalPopulation = trimPopulation(newPopulation, popSize)
    return finalPopulation


def geneticAlgorithm(cityList, popSize, numPairs, mutationRate, generations, tol=1e-4, convergenceLimit=5):

    # Step 1: Initialize population
    population = initialPolpulation(popSize, cityList)

    bestDistances = []
    avgFitnesses = []

    for gen in range(generations):
        # Step 2: Generate next generation
        population = nextGeneration(population, numPairs, popSize, mutationRate)

        # Step 3: Track best and average fitness
        fitnesses = [Fitness(route).fitness for route in population]
        avgFitness = sum(fitnesses) / len(fitnesses)
        bestFitness = max(fitnesses)
        bestRoute = population[fitnesses.index(bestFitness)]
        bestDistance = 1 / bestFitness

        bestDistances.append(bestDistance)
        avgFitnesses.append(avgFitness)

        # Print progress
        if gen % 10 == 0:
            print(f"Generation {gen}: Best distance = {bestDistance:.2f}, Avg fitness = {avgFitness:.5f}")
            Plot(bestRoute, bestDistance)

    
    # Plot fitness progression
    plt.figure(figsize=(10,5))
    plt.plot(avgFitnesses, label="Average Fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.title("Fitness Evolution")
    plt.legend()
    plt.show()

    Plot(bestRoute, bestDistance)

    return bestRoute, bestDistances


if __name__ == "__main__":
    cityList = readCities()
    bestRoute, bestDistance = geneticAlgorithm(cityList, popSize=100, numPairs=20, mutationRate=0.08, generations=200)
