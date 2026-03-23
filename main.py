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

"""
print("\nTop 10 de la population initiale :")
for mot, score in scores_population[:10]:
    print(f"{mot}: {score}")"""


ga = ga( population_size=20,mutation_rate=0.2,trigram_model=trigram_model,dictionary_set=dictionary_set, choice_indiv=2, crossover_type='one_point')


best_word, best_score = ga.run(nb_generations=50)



eda = eda(population_size=100,mutation_rate=0.05,trigram_model=trigram_model,dictionary_set=dictionary_set,choice_indiv=2)

best_word, best_score = eda.run(nb_generations=50,nombre_prts=20)

print("Final best word :", best_word)
print("Final best score :", best_score)