from population import init_population
from evaluation import fonction_objective
import random


class eda:

    def __init__(self, population_size, perturbation_aleatoire, trigram_model, dictionary_set, choice_indiv):
        self.population_size = population_size
        self.perturbation_aleatoire = perturbation_aleatoire
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

        taille_mot = max(len(mot) for mot in parents)
        #valeur_lissage_test = 0.05

        for i in range(taille_mot):
            compt = {}

            for letter in self.alphabet:
                compt[letter] = 1

            total =  len(self.alphabet)

            for mot in parents:
                if i < len(mot):
                    letter = mot[i]
                    compt[letter] += 1
                    total += 1

            proba = {}
            for letter in self.alphabet:
                proba[letter] = compt[letter] / total

            distribution.append(proba)

        return distribution
    
    def estimate_length_distribution(self, parents):
        length_count = {}

        for mot in parents:
            longueur = len(mot)
            if longueur not in length_count:
                length_count[longueur] = 0
            length_count[longueur] += 1

        total = len(parents)
        length_proba = {}

        for longueur in length_count:
            length_proba[longueur] = length_count[longueur] / total

        return length_proba
    
    def choose_length(self, length_proba):
        longueurs = list(length_proba.keys())
        probs = list(length_proba.values())

        longueur = random.choices(longueurs, weights=probs, k=1)[0]
        return longueur
    
    def choose_letter(self, probabilites_position):
        lettres = list(probabilites_position.keys())
        probs = list(probabilites_position.values())

        lettre = random.choices(lettres, weights=probs, k=1)[0]
        return lettre

    def create_word(self, distribution, longueur):
        mot_temp = ""

        for i in range(longueur):
            if i < len(distribution):
                letter = self.choose_letter(distribution[i])
            else:
                letter = random.choice(self.alphabet)

            if random.random() < self.perturbation_aleatoire:
                letter = random.choice(self.alphabet)

            mot_temp += letter

        return mot_temp
    
    def create_new_population(self, distribution, scores, length_proba):
        new_population = []

        best_words = [mot for mot, _ in scores[:self.choice_indiv]]
        for word in best_words:
            new_population.append(word)

        while len(new_population) < self.population_size:
            longueur = self.choose_length(length_proba)
            new_word = self.create_word(distribution, longueur)
            new_population.append(new_word)

        return new_population

    def run(self, nb_generations, nombre_prts):

        history = []
        best_word = None
        best_score = float("inf")
        compteur_fonction_objective = 0

        for generation in range(nb_generations):
            scores = self.evaluate_population()
            compteur_fonction_objective += len(self.population)
            current_word = scores[0][0]
            current_score = scores[0][1]

            if current_score < best_score:
                best_word = current_word
                best_score = current_score
            
            moyenne_result = sum(score for _, score in scores) / len(scores)

            parents = self.parent_selection(scores, nombre_prts)

            distribution = self.distribution_estimation(parents)
            length_proba = self.estimate_length_distribution(parents)

            self.population = self.create_new_population(distribution, scores, length_proba)

            history.append({"generation": generation,"best_word": current_word,"best_score": current_score,"average_score": moyenne_result,"diversity": len(set(self.population))})

        return best_word, best_score, history , compteur_fonction_objective