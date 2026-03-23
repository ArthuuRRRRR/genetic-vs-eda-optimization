#%%

import numpy as np
import nltk
from nltk.util import pad_sequence
from nltk.lm.preprocessing import pad_both_ends
from nltk.lm.preprocessing import flatten
from nltk import FreqDist, ConditionalFreqDist
from collections import defaultdict

import gen_lm
import generate_corpus
from evaluation import fonction_objective
from population import init_population
from ga import ga
from eda import eda
from run_monte_carlo import monte_carlo_ga, monte_carlo_eda, run_ga_simple, run_eda_simple

#%%
# voici une brève démonstration de l'utilisation du code fourni

dictionnaire = generate_corpus.generate_dictionary() # peut prendre un peu de temps
dictionary_set = set(dictionnaire) 

# dans notre cas, on va bâtir bêtement le corpus directement à partir du dictionnaire traité
corpus_entrainement = dictionnaire


trigram_model = gen_lm.build_trigram_model(corpus_entrainement)

mots = ['bonjour', 'jourbon', 'manger', 'aaaaa', 'allo']


# %%
for mot in mots:
    ppl = gen_lm.perplexité(mot=mot, trigram_model=trigram_model)
    print(f"{mot}: {ppl}")

# %%

# %%
print("\nTest de la fonction objective :")
for mot in mots:
    score = fonction_objective(trigram_model, mot, dictionary_set)
    print(f"{mot}: {score}")

# %%
population = init_population(20)

scores_population = []
for mot in population:
    score = fonction_objective(trigram_model, mot, dictionary_set)
    scores_population.append((mot, score))

scores_population = sorted(scores_population, key=lambda x: x[1])

results_ga = monte_carlo_ga(n_runs=5,nb_generations=10,population_size=20,mutation_rate=0.2,trigram_model=trigram_model,dictionary_set=dictionary_set,choice_indiv=2,crossover_type="one_point")

print("\n--- Résultats finaux GA ---")
for result in results_ga:
    print("seed =", result["seed"],"| best_word =", result["best_word"],"| best_score =", result["best_score"])


results_eda = monte_carlo_eda(n_runs=5,nb_generations=10,nombre_prts=10,population_size=20,mutation_rate=0.2,trigram_model=trigram_model,dictionary_set=dictionary_set,choice_indiv=2)

print("\n--- Résultats finaux EDA ---")
for result in results_eda:
    print("seed =", result["seed"],"| best_word =", result["best_word"],"| best_score =", result["best_score"])