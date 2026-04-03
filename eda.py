from population import init_population
from evaluation import fonction_objective
import random


class eda:

    def __init__(self, population_size, perturbation_aleatoire, trigram_model, dictionary_set, choice_indiv):
        """
        Cette classe génère une nouvelle population en estimant les distributions de probabilités (lettres et longueurs) 
        à partir des meilleurs individus, puis en échantillonnant de nouveaux mots selon ces distributions. 
        
        Paramètres principaux : 
        - population_size : taille de la population 
        - perturbation_aleatoire : probabilité d ajouter du bruit aléatoire 
        - choice_indiv : nombre de meilleurs individus conservés
        
        """
        self.population_size = population_size
        self.perturbation_aleatoire = perturbation_aleatoire
        self.trigram_model = trigram_model
        self.dictionary_set = dictionary_set
        self.population = init_population(population_size)

        self.choice_indiv = choice_indiv
        self.alphabet = "abcdefghijklmnopqrstuvwxyz"
    
    def evaluate_population(self): # je calcule le score de chaque individu et je retourne une liste triée (mot, score)
        results = []

        for mot in self.population:
            result = fonction_objective(self.trigram_model, mot, self.dictionary_set)
            results.append((mot, result))

        return sorted(results, key=lambda x: x[1])
    
    def parent_selection(self, scores, nombre_prts): # sélectionne les meilleurs individus selon leur score
        return [mot for mot, _ in scores[:nombre_prts]]
    
    def distribution_estimation(self, parents): #  estime une distribution de probabilité des lettres à chaque position
        distribution = []

        if len(parents) == 0:
            return distribution

        taille_mot = max(len(mot) for mot in parents) # prends la longueur maximale pour modéliser toutes les positions
        #valeur_lissage_test = 0.05

        for i in range(taille_mot): # Pour chaque position, je compte les occurrences des lettres
            compt = {}

            for letter in self.alphabet:
                compt[letter] = 1

            total =  len(self.alphabet) 

            for mot in parents: # je regarde ensuite la lettre à la position i et j ajoute 1 au comptage de cette lettre
                if i < len(mot):
                    letter = mot[i]
                    compt[letter] += 1
                    total += 1

            proba = {}
            for letter in self.alphabet: # Je normalise pour obtenir une distribution de probabilité
                proba[letter] = compt[letter] / total

            distribution.append(proba) 

        return distribution
    
    def estimate_length_distribution(self, parents): # J’estime une distribution de probabilité des longueurs de mots
        length_count = {}

        for mot in parents:     #  compte les occurrences de chaque longueur
            longueur = len(mot)
            if longueur not in length_count:
                length_count[longueur] = 0
            length_count[longueur] += 1

        total = len(parents)
        length_proba = {}

        for longueur in length_count:
            length_proba[longueur] = length_count[longueur] / total # normalise pour obtenir les probabilités

        return length_proba
    
    def choose_length(self, length_proba): # tire une longueur selon la distribution estimée
        longueurs = list(length_proba.keys())
        probs = list(length_proba.values())

        longueur = random.choices(longueurs, weights=probs, k=1)[0]
        return longueur
    
    def choose_letter(self, probabilites_position): # tire une lettre selon la distribution à une position donnée
        lettres = list(probabilites_position.keys())
        probs = list(probabilites_position.values())

        lettre = random.choices(lettres, weights=probs, k=1)[0]
        return lettre

    def create_word(self, distribution, longueur): # Je génère un mot en échantillonnant chaque position
        mot_temp = ""

        for i in range(longueur):
            if i < len(distribution):
                letter = self.choose_letter(distribution[i])
            else:
                letter = random.choice(self.alphabet)

            if random.random() < self.perturbation_aleatoire: # J’ajoute du bruit pour maintenir la diversité
                letter = random.choice(self.alphabet)

            mot_temp += letter

        return mot_temp
    
    def create_new_population(self, distribution, scores, length_proba): # génère nouvelle population en échantillonnant de nouveaux mots selon les distributions estimées à partir des meilleurs individus
        new_population = []

        best_words = [mot for mot, _ in scores[:self.choice_indiv]] # Je conserve les meilleurs individus (élitisme)
        for word in best_words:
            new_population.append(word)

        while len(new_population) < self.population_size:
            longueur = self.choose_length(length_proba)
            new_word = self.create_word(distribution, longueur)
            new_population.append(new_word)

        return new_population

    def run(self, nb_generations, nombre_prts): # fais évoluer la population sur plusieurs générations

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