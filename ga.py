from population import init_population
from evaluation import fonction_objective


class ga:
    def __init__(self, population_size, mutation_rate):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = init_population(population_size)
    
    def evaluate_population(self):
        results = []
        for mot in self.population:
            result = fonction_objective(self.trigram_model, mot, self.dictionary_set)
            results.append((mot, result))
        return sorted(results, key=lambda x: x[1])  

    def parent_selection(self, scores, nombre_prts):
        return [mot for mot, _ in scores[:nombre_prts]]

    def crossover(self, parent1, parent2):
        pass

    def mutation(self, individual):
        pass


    
    def run(self):
        evaluate=self.evaluate_population() 
        pass
    
    
