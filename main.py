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
from display_result import plot_convergence, plot_average_population, plot_boxplot, print_summary, print_best_runs, plot_sorted_final_scores

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
"""
print("\nTest de la fonction objective :")
for mot in mots:
    score = fonction_objective(trigram_model, mot, dictionary_set)
    print(f"{mot}: {score}")
"""
# %%


population = init_population(50)

print("Que voulez vous faire ?")
print("1. Faire une comparaison entre GA et EDA")
print("2. Faire une analyse sur GA")
print("3. Faire une analyse sur EDA")
choice = input("Entrez votre choix (1, 2 ou 3) : ")


if choice == "1":
    results_ga = monte_carlo_ga(n_runs=20,nb_generations=40,population_size=60,mutation_rate_pm=0.2,trigram_model=trigram_model,dictionary_set=dictionary_set,crossover_type="one_point",etalon=True,elitisme=2,losers=2)

    results_eda = monte_carlo_eda(n_runs=20,nb_generations=40,nombre_prts=10,population_size=60,mutation_rate=0.2,trigram_model=trigram_model,dictionary_set=dictionary_set,choice_indiv=2)
    
    plot_convergence(results_ga, results_eda)
    plot_average_population(results_ga, results_eda)
    plot_boxplot(results_ga, results_eda)
    plot_sorted_final_scores(results_ga, results_eda)
    
    print_summary(results_ga, "GA")
    print_best_runs(results_ga, "GA")
    
    print_summary(results_eda, "EDA")
    print_best_runs(results_eda, "EDA")


elif choice == "2":

    print("\n--- Comparaison crossover ---")
    results_ga_1 = monte_carlo_ga(n_runs=20,nb_generations=40,population_size=60,mutation_rate_pm=0.2,trigram_model=trigram_model,dictionary_set=dictionary_set,crossover_type="one_point")

    results_ga_2 = monte_carlo_ga(n_runs=20,nb_generations=40,population_size=60,mutation_rate_pm=0.2,trigram_model=trigram_model,dictionary_set=dictionary_set,crossover_type="uniform")

    plot_convergence(results_ga_1, results_ga_2, "one_point", "uniform")
    plot_average_population(results_ga_1, results_ga_2, "one_point", "uniform")
    plot_boxplot(results_ga_1, results_ga_2, "one_point", "uniform")
    plot_sorted_final_scores(results_ga_1, results_ga_2, "one_point", "uniform")

    print("\n--- Effet de l'étalon ---")
    ga_sans_etalon = monte_carlo_ga(n_runs=20,nb_generations=40,population_size=60,mutation_rate_pm=0.2,trigram_model=trigram_model,dictionary_set=dictionary_set,etalon=False)

    ga_avec_etalon = monte_carlo_ga(n_runs=20,nb_generations=40,population_size=60,mutation_rate_pm=0.2,trigram_model=trigram_model,dictionary_set=dictionary_set,etalon=True)

    plot_convergence(ga_sans_etalon, ga_avec_etalon, "sans étalon", "avec étalon")
    plot_average_population(ga_sans_etalon, ga_avec_etalon, "sans étalon", "avec étalon")
    plot_boxplot(ga_sans_etalon, ga_avec_etalon, "sans étalon", "avec étalon")
    plot_sorted_final_scores(ga_sans_etalon, ga_avec_etalon, "sans étalon", "avec étalon")


    print("\n--- Effet des losers ---")
    ga_sans_losers = monte_carlo_ga(n_runs=20,nb_generations=40,population_size=60,mutation_rate_pm=0.2,trigram_model=trigram_model,dictionary_set=dictionary_set,losers=0)

    ga_avec_losers = monte_carlo_ga(n_runs=20,nb_generations=40,population_size=60,mutation_rate_pm=0.2,trigram_model=trigram_model,dictionary_set=dictionary_set,losers=3)

    plot_convergence(ga_sans_losers, ga_avec_losers, "sans losers", "avec losers")
    plot_average_population(ga_sans_losers, ga_avec_losers, "sans losers", "avec losers")
    plot_boxplot(ga_sans_losers, ga_avec_losers, "sans losers", "avec losers")
    plot_sorted_final_scores(ga_sans_losers, ga_avec_losers, "sans losers", "avec losers")


elif choice == "3":

    print("\n--- Effet du nombre de parents (EDA) ---")

    results_eda_1 = monte_carlo_eda(n_runs=20,nb_generations=40,nombre_prts=10,population_size=60,mutation_rate=0.2,trigram_model=trigram_model,dictionary_set=dictionary_set,choice_indiv=2)

    results_eda_2 = monte_carlo_eda(n_runs=20,nb_generations=40,nombre_prts=20,population_size=60,mutation_rate=0.2,trigram_model=trigram_model,dictionary_set=dictionary_set,choice_indiv=2)

    plot_convergence(results_eda_1, results_eda_2, "parents=10", "parents=20")
    plot_average_population(results_eda_1, results_eda_2, "parents=10", "parents=20")
    plot_boxplot(results_eda_1, results_eda_2, "parents=10", "parents=20")
    plot_sorted_final_scores(results_eda_1, results_eda_2, "parents=10", "parents=20")

