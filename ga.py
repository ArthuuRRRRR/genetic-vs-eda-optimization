from population import init_population
from evaluation import fonction_objective
import random


class ga:
    def __init__(self,population_size,mutation_rate_pm,trigram_model,dictionary_set,choice_indiv=2,crossover_type='one_point',min_length=4,max_length=16,crossover_rate_pc=0.8,reseed=2,elitisme=2,etalon=False,losers=0):
        self.population_size = population_size
        self.mutation_rate_pm = mutation_rate_pm
        self.trigram_model = trigram_model
        self.dictionary_set = dictionary_set
        self.crossover_type = crossover_type
        self.population = init_population(population_size)
        self.choice_indiv = choice_indiv
        self.alphabet = "abcdefghijklmnopqrstuvwxyz"
        self.min_length = min_length
        self.max_length = max_length

        self.crossover_rate_pc = crossover_rate_pc
        self.reseed = reseed

        self.elitisme = elitisme
        self.etalon = etalon
        self.losers = losers
    
    def evaluate_population(self):
        results = []
        for mot in self.population:
            result = fonction_objective(self.trigram_model, mot, self.dictionary_set)
            results.append((mot, result))
        return sorted(results, key=lambda x: x[1])

    def parent_selection(self, scores, nombre_prts):
        return [mot for mot, _ in scores[:nombre_prts]]

    def crossover_one_point(self, parent1, parent2):
        if len(parent1) < 2 or len(parent2) < 2:
            return parent1, parent2

        stop1 = random.randint(1, len(parent1) - 1)
        stop2 = random.randint(1, len(parent2) - 1)

        child1 = parent1[:stop1] + parent2[stop2:]
        child2 = parent2[:stop2] + parent1[stop1:]
        return child1, child2
    
    def crossover_uniform(self, parent1, parent2):
        minim = min(len(parent1), len(parent2))
        
        child1 = []
        child2 = []
        for i in range(minim):
            if random.random() < 0.5:
                child1.append(parent1[i])
                child2.append(parent2[i])
            else:
                child1.append(parent2[i])
                child2.append(parent1[i])
        
        if len(parent1) > minim:
            child1.extend(parent1[minim:])
        if len(parent2) > minim:
            child2.extend(parent2[minim:])
        return ''.join(child1), ''.join(child2)
    
    def crossover(self, parent1, parent2):
        if self.crossover_type == "uniform":
            return self.crossover_uniform(parent1, parent2)
        else:
            return self.crossover_one_point(parent1, parent2)

    def mutation(self, word):
        if random.random() >= self.mutation_rate_pm:
            return word

        operation = random.choice(["replace", "insert", "delete"])

        if operation == "replace" and len(word) > 0:
            i = random.randint(0, len(word) - 1)
            c = random.choice(self.alphabet)
            return word[:i] + c + word[i+1:]

        if operation == "insert" and len(word) < self.max_length:
            i = random.randint(0, len(word))
            c = random.choice(self.alphabet)
            return word[:i] + c + word[i:]

        if operation == "delete" and len(word) > self.min_length:
            i = random.randint(0, len(word) - 1)
            return word[:i] + word[i+1:]

        return word
    
    def run(self, nb_generations):
        best_word = None
        best_score = float("inf")
        history = []
        compteur_fonction_objective = 0

        for generation in range(nb_generations):
            results = self.evaluate_population()
            compteur_fonction_objective += len(self.population)

            current_word = results[0][0]
            current_score = results[0][1]

            if current_score < best_score:
                best_word = current_word
                best_score = current_score

            moyenne_result = sum(score for _, score in results) / len(results)

            best_words = [mot for mot, _ in results[:self.elitisme]]
            new_population = best_words.copy()

            if self.losers > 0:
                worst_words = [mot for mot, _ in results[-self.losers:]]
                new_population.extend(worst_words)

            nombre_parents = max(2, self.population_size // 2)
            parents = self.parent_selection(results, nombre_parents)

            if self.etalon:
                stallion = results[0][0]

            while len(new_population) < self.population_size:

                if self.etalon and random.random() < 0.5:
                    parent1 = stallion
                    parent2 = random.choice(parents)
                else:
                    parent1, parent2 = random.sample(parents, 2)

                if random.random() < self.crossover_rate_pc:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2

                child1 = self.mutation(child1)
                child2 = self.mutation(child2)

                new_population.append(child1)

                if len(new_population) < self.population_size:
                    new_population.append(child2)

            if self.reseed > 0:
                for _ in range(min(self.reseed, len(new_population))):
                    if len(new_population) > self.elitisme:
                        idx = random.randint(self.elitisme, len(new_population) - 1)
                    else:
                        idx = random.randint(0, len(new_population) - 1)

                    new_population[idx] = init_population(1)[0]
            
            self.population = new_population[:self.population_size]

            history.append({"generation": generation,"best_word": current_word,"best_score": current_score,"average_score": moyenne_result,"diversity": len(set(self.population))})

        return best_word, best_score, history , compteur_fonction_objective