from population import init_population


class ga:
    def __init__(self, population_size, mutation_rate):
        self.population_size = population_size
        self.mutation_rate = mutation_rate

        self.population = init_population(population_size)
    
    def evaluate_population(self):
        pass

    
    def ga (self):
        evaluate=self.evaluate_population() 
        pass
    
    
