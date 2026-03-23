from population import init_population
from evaluation import fonction_objective
import random


class eda:

    def __init__(self, population_size, mutation_rate, trigram_model, dictionary_set, choice_indiv):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.trigram_model = trigram_model
        self.dictionary_set = dictionary_set
        self.population = init_population(population_size)

        self.choice_indiv = choice_indiv
        self.alphabet = "abcdefghijklmnopqrstuvwxyz"
    
    def evaluate_population(self):
        results = []

        for mot in self.population:
            result = fonction_objective(self.trigram_model, mot, self.dictionary_set)
            results.append((mot, result))

        return sorted(results, key=lambda x: x[1])
    
    def parent_selection(self, scores, nombre_prts):
        return [mot for mot, _ in scores[:nombre_prts]]
    
    def distribution_estimation(self, parents):
        distribution = []

        if len(parents) == 0:
            return distribution

        taille_mot = min(len(mot) for mot in parents)

        for i in range(taille_mot):
            compt = {}

            for letter in self.alphabet:
                compt[letter] = 0

            for mot in parents:
                letter = mot[i]
                compt[letter] += 1

            proba = {}
            for letter in self.alphabet:
                proba[letter] = compt[letter] / len(parents)

            distribution.append(proba)

        return distribution
    
    def choose_letter(self, probabilites_position):
        lettres = list(probabilites_position.keys())
        probs = list(probabilites_position.values())

        lettre = random.choices(lettres, weights=probs, k=1)[0]
        return lettre

    def create_word(self, distribution):
        mot_temp = ""

        for probabilites_position in distribution:
            letter = self.choose_letter(probabilites_position)

            if random.random() < self.mutation_rate:
                letter = random.choice(self.alphabet)

            mot_temp += letter

        return mot_temp
    
    def create_new_population(self, distribution, scores):
        new_population = []

        best_words = [mot for mot, _ in scores[:self.choice_indiv]]
        for word in best_words:
            new_population.append(word)

        while len(new_population) < self.population_size:
            new_word = self.create_word(distribution)
            new_population.append(new_word)

        return new_population

    def run(self, nb_generations, nombre_prts):


        best_word = None
        best_score = float("inf")

        for generation in range(nb_generations):
            scores = self.evaluate_population()
            current_word = scores[0][0]
            current_score = scores[0][1]

            if current_score < best_score:
                best_word = current_word
                best_score = current_score

            print("Generation :", generation + 1)
            print("Best word :", current_word)
            print("Score :", current_score)

            parents = self.parent_selection(scores, nombre_prts)

            distribution = self.distribution_estimation(parents)
            self.population = self.create_new_population(distribution, scores)

        return best_word, best_score