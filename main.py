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
from display_result import plot_convergence, plot_average_population, plot_violin, print_summary, print_best_runs, plot_sorted_final_scores, plot_diversity
from annexe_words import generate_annexe

import argparse

#%%
# voici une brève démonstration de l'utilisation du code fourni

dictionnaire = generate_corpus.generate_dictionary() # peut prendre un peu de temps
dictionary_set = set(dictionnaire) 

# dans notre cas, on va bâtir bêtement le corpus directement à partir du dictionnaire traité
corpus_entrainement = dictionnaire


trigram_model = gen_lm.build_trigram_model(corpus_entrainement)


# %%
"""
Par défaut, si aucun argument n est fourni en ligne de commande, j utilise un menu interactif
qui permet de lancer différentes analyses (comparaison GA vs EDA, analyse détaillée, génération d annexes, etc...).
Ce mode est pratique pour explorer rapidement les résultats sans avoir à retenir les paramètres.

J ai aussi ajouté un second mode basé sur argparse qui permet de lancer directement
un algorithme (GA ou EDA) depuis la ligne de commande en spécifiant les paramètres.
Cela permet de modifier les paramètres sans toucher au code. 

les commandes sont : 
python main.py
python main.py --algorithm ga --crossover_type uniform --etalon --elitisme 2 --losers 0 --reseed 2 --population_size 60 --nb_generations 40 --mutation_rate 0.2
python main.py --algorithm eda --population_size 60 --nb_generations 40 --perturbation_rate 0.2 --nombre_parents 20 --choice_indiv 2



"""


def menu():
    population = init_population(60)

    print("Que voulez vous faire ?")
    print("1. Faire une comparaison entre GA et EDA")
    print("2. Faire une analyse sur GA")
    print("3. Faire une analyse sur EDA")
    print("4. Generer annexes")
    print("5. Faire les runs avec paramètres entrés manuellement")
    choice = input("Entrez votre choix (1, 2, 3, 4 ou 5) : ")


    if choice == "1":
        results_ga = monte_carlo_ga(n_runs=20,nb_generations=40,population_size=60,mutation_rate_pm=0.2,trigram_model=trigram_model,dictionary_set=dictionary_set,crossover_type="one_point",etalon=True,elitisme=2,losers=0)

        results_eda = monte_carlo_eda(n_runs=20,nb_generations=40,nombre_prts=5,population_size=60,perturbation_aleatoire=0.2,trigram_model=trigram_model,dictionary_set=dictionary_set,choice_indiv=2)
        
        plot_convergence(results_ga, results_eda, "GA", "EDA")
        plot_average_population(results_ga, results_eda, "GA", "EDA")
        plot_violin(results_ga, results_eda, "GA", "EDA")
        plot_sorted_final_scores(results_ga, results_eda, "GA", "EDA")
        plot_diversity(results_ga, results_eda, "GA", "EDA")

        print_summary(results_ga, "GA")
        print_best_runs(results_ga, "GA")
        
        print_summary(results_eda, "EDA")
        print_best_runs(results_eda, "EDA")

        print("nombre de evaluations de la fonction objectif pour EDA : ", sum(r["compteur_fonction_objective"] for r in results_eda))
        print("nombre de evaluations de la fonction objectif pour GA : ", sum(r["compteur_fonction_objective"] for r in results_ga))


    elif choice == "2":

        print("\n--- Comparaison crossover ---")
        results_ga_1 = monte_carlo_ga(n_runs=20,nb_generations=40,population_size=60,mutation_rate_pm=0.2,trigram_model=trigram_model,dictionary_set=dictionary_set,crossover_type="one_point")

        results_ga_2 = monte_carlo_ga(n_runs=20,nb_generations=40,population_size=60,mutation_rate_pm=0.2,trigram_model=trigram_model,dictionary_set=dictionary_set,crossover_type="uniform")

        plot_convergence(results_ga_1, results_ga_2, "one_point", "uniform")
        plot_average_population(results_ga_1, results_ga_2, "one_point", "uniform")
        plot_violin(results_ga_1, results_ga_2, "one_point", "uniform")
        plot_sorted_final_scores(results_ga_1, results_ga_2, "one_point", "uniform")
        plot_diversity(results_ga_1, results_ga_2, "one_point", "uniform")
        print_summary(results_ga_1, "one_point")
        print_best_runs(results_ga_1, "one_point")
        print_summary(results_ga_2, "uniform")
        print_best_runs(results_ga_2, "uniform")

        print("\n--- Effet de l'étalon ---")
        ga_sans_etalon = monte_carlo_ga(n_runs=20,nb_generations=40,population_size=60,mutation_rate_pm=0.2,trigram_model=trigram_model,dictionary_set=dictionary_set,etalon=False)

        ga_avec_etalon = monte_carlo_ga(n_runs=20,nb_generations=40,population_size=60,mutation_rate_pm=0.2,trigram_model=trigram_model,dictionary_set=dictionary_set,etalon=True)

        plot_convergence(ga_sans_etalon, ga_avec_etalon, "sans étalon", "avec étalon")
        plot_average_population(ga_sans_etalon, ga_avec_etalon, "sans étalon", "avec étalon")
        plot_violin(ga_sans_etalon, ga_avec_etalon, "sans étalon", "avec étalon")
        plot_sorted_final_scores(ga_sans_etalon, ga_avec_etalon, "sans étalon", "avec étalon")
        plot_diversity(ga_sans_etalon, ga_avec_etalon, "sans étalon", "avec étalon")
        print_summary(ga_sans_etalon, "sans étalon")
        print_best_runs(ga_sans_etalon, "sans étalon")
        print_summary(ga_avec_etalon, "avec étalon")
        print_best_runs(ga_avec_etalon, "avec étalon")


        print("\n--- Effet des losers ---")
        ga_sans_losers = monte_carlo_ga(n_runs=20,nb_generations=40,population_size=60,mutation_rate_pm=0.2,trigram_model=trigram_model,dictionary_set=dictionary_set,losers=0)

        ga_avec_losers = monte_carlo_ga(n_runs=20,nb_generations=40,population_size=60,mutation_rate_pm=0.2,trigram_model=trigram_model,dictionary_set=dictionary_set,losers=3)

        plot_convergence(ga_sans_losers, ga_avec_losers, "sans losers", "avec losers")
        plot_average_population(ga_sans_losers, ga_avec_losers, "sans losers", "avec losers")
        plot_violin(ga_sans_losers, ga_avec_losers, "sans losers", "avec losers")
        plot_sorted_final_scores(ga_sans_losers, ga_avec_losers, "sans losers", "avec losers")
        plot_diversity(ga_sans_losers, ga_avec_losers, "sans losers", "avec losers")
        print_summary(ga_sans_losers, "sans losers")
        print_best_runs(ga_sans_losers, "sans losers")
        print_summary(ga_avec_losers, "avec losers")
        print_best_runs(ga_avec_losers, "avec losers")

        print("\n--- Effet du reseeds---")
        ga_sans_reseed = monte_carlo_ga(n_runs=20,nb_generations=40,population_size=60,mutation_rate_pm=0.2,trigram_model=trigram_model,dictionary_set=dictionary_set,reseed=0)
        ga_avec_reseed = monte_carlo_ga(n_runs=20,nb_generations=40,population_size=60,mutation_rate_pm=0.2,trigram_model=trigram_model,dictionary_set=dictionary_set,reseed=5)

        plot_convergence(ga_sans_reseed, ga_avec_reseed, "sans reseed", "avec reseed")
        plot_average_population(ga_sans_reseed, ga_avec_reseed, "sans reseed", "avec reseed")
        plot_violin(ga_sans_reseed, ga_avec_reseed, "sans reseed", "avec reseed")
        plot_sorted_final_scores(ga_sans_reseed, ga_avec_reseed, "sans reseed", "avec reseed")
        plot_diversity(ga_sans_reseed, ga_avec_reseed, "sans reseed", "avec reseed")
        print_summary(ga_sans_reseed, "sans reseed")
        print_best_runs(ga_sans_reseed, "sans reseed")
        print_summary(ga_avec_reseed, "avec reseed")
        print_best_runs(ga_avec_reseed, "avec reseed")


    elif choice == "3":

        print("\n--- Effet du nombre de parents (EDA) ---")

        results_eda_1 = monte_carlo_eda(n_runs=20,nb_generations=40,nombre_prts=10,population_size=60,perturbation_aleatoire=0.2,trigram_model=trigram_model,dictionary_set=dictionary_set,choice_indiv=2)

        results_eda_2 = monte_carlo_eda(n_runs=20,nb_generations=40,nombre_prts=20,population_size=60,perturbation_aleatoire=0.2,trigram_model=trigram_model,dictionary_set=dictionary_set,choice_indiv=2)

        plot_convergence(results_eda_1, results_eda_2, "parents=10", "parents=20")
        plot_average_population(results_eda_1, results_eda_2, "parents=10", "parents=20")
        plot_violin(results_eda_1, results_eda_2, "parents=10", "parents=20")
        plot_sorted_final_scores(results_eda_1, results_eda_2, "parents=10", "parents=20")
        plot_diversity(results_eda_1, results_eda_2, "parents=10", "parents=20")
        print_summary(results_eda_1, "parents=10")
        print_best_runs(results_eda_1, "parents=10")
        print_summary(results_eda_2, "parents=20")
        print_best_runs(results_eda_2, "parents=20")


        print("\n--- Effet du choix des individus (EDA) ---")

        results_eda_3 = monte_carlo_eda(n_runs=20,nb_generations=40,nombre_prts=20,population_size=60,perturbation_aleatoire=0.2,trigram_model=trigram_model,dictionary_set=dictionary_set,choice_indiv=2)

        results_eda_4 = monte_carlo_eda(n_runs=20,nb_generations=40,nombre_prts=20,population_size=60,perturbation_aleatoire=0.2,trigram_model=trigram_model,dictionary_set=dictionary_set,choice_indiv=10)

        plot_convergence(results_eda_3, results_eda_4, "choix_indiv=2", "choix_indiv=10")
        plot_average_population(results_eda_3, results_eda_4, "choix_indiv=2", "choix_indiv=10")
        plot_violin(results_eda_3, results_eda_4, "choix_indiv=2", "choix_indiv=10")
        plot_sorted_final_scores(results_eda_3, results_eda_4, "choix_indiv=2", "choix_indiv=10")
        plot_diversity(results_eda_3, results_eda_4, "choix_indiv=2", "choix_indiv=10")
        print_summary(results_eda_3, "choix_indiv=2")
        print_best_runs(results_eda_3, "choix_indiv=2")
        print_summary(results_eda_4, "choix_indiv=10")
        print_best_runs(results_eda_4, "choix_indiv=10")
                                

    elif choice == "4":

        print("\n--- Génération annexe ---")

        results_ga = monte_carlo_ga(
            n_runs=80,
            nb_generations=100,
            population_size=90,
            mutation_rate_pm=0.2,
            trigram_model=trigram_model,
            dictionary_set=dictionary_set
        )

        results_eda = monte_carlo_eda(
            n_runs=80,
            nb_generations=100,
            nombre_prts=20,
            population_size=90,
            perturbation_aleatoire=0.2,
            trigram_model=trigram_model,
            dictionary_set=dictionary_set,
            choice_indiv=2
        )

        generate_annexe(results_ga, trigram_model, dictionary_set, "annexe_ga1.csv")
        generate_annexe(results_eda, trigram_model, dictionary_set, "annexe_eda1.csv")

    elif choice == "5":
        print("\n--- Runs avec paramètres entrés manuellement ---")
        nbr_population = int(input("Entrez la taille de la population : "))
        seed = int(input("Entrez la seed : "))
        nb_generations = int(input("Entrez le nombre de générations : "))
        population_size = int(input("Entrez la taille de la population : "))
        mutation_rate_pm = float(input("Entrez le taux de mutation (pour GA) : "))
        perturbation_aleatoire = float(input("Entrez le taux de perturbation pour la robustesse(pour EDA) : "))
        nombre_prts = int(input("Entrez le nombre de parents (pour EDA) : "))
        choice_indiv = int(input("Entrez le choix des individus (pour EDA) : "))    

        population = init_population(nbr_population)

        result_ga = run_ga_simple(seed=seed,nb_generations=nb_generations,population_size=population_size,mutation_rate_pm=mutation_rate_pm,trigram_model=trigram_model,dictionary_set=dictionary_set,crossover_type="one_point",etalon=True,elitisme=2, losers=0, reseed=2)
        result_eda = run_eda_simple(seed=seed,nb_generations=nb_generations,nombre_prts=nombre_prts,population_size=population_size,perturbation_aleatoire=perturbation_aleatoire,trigram_model=trigram_model,dictionary_set=dictionary_set,choice_indiv=choice_indiv)

        print("\nRésultats GA :")
        print_summary([result_ga], "GA")
        print_best_runs([result_ga], "GA")

        print("\nRésultats EDA :")
        print_summary([result_eda], "EDA")
        print_best_runs([result_eda], "EDA")

def argparse_menu():
    parser = argparse.ArgumentParser(description="Run GA ou EDA avec paramètres manuel")
    parser.add_argument("--algorithm", choices=["ga", "eda"], help="Choix de l'algorithme à exécuter (ga ou eda)")
    parser.add_argument("--crossover_type", choices=["one_point", "uniform"], default="one_point")
    parser.add_argument("--etalon", action="store_true")
    parser.add_argument("--elitisme", type=int, default=2)
    parser.add_argument("--losers", type=int, default=0)
    parser.add_argument("--reseed", type=int, default=2)
    parser.add_argument("--population_size", type=int, default=60, help="Taille de la population")
    parser.add_argument("--nb_generations", type=int, default=40, help="Nombre de générations")
    parser.add_argument("--mutation_rate", type=float, default=0.2, help="Taux de mutation pour GA")
    parser.add_argument("--perturbation_rate", type=float, default=0.2, help="Taux de perturbation pour EDA")
    parser.add_argument("--nombre_parents", type=int, default=20, help="Nombre de parents pour EDA")
    parser.add_argument("--choice_indiv", type=int, default=2, help="Choix des individus pour EDA")
    parser.add_argument("--seed", type=int, default=42, help="Seed pour la reproductibilité")
    args = parser.parse_args()
    return args


def main():
    args = argparse_menu()

    if args.algorithm == "ga":
        result_ga = run_ga_simple(
            seed=args.seed,
            crossover_type=args.crossover_type,
            etalon=args.etalon,
            elitisme=args.elitisme,
            losers=args.losers,
            reseed=args.reseed,
            nb_generations=args.nb_generations,
            population_size=args.population_size,
            mutation_rate_pm=args.mutation_rate,
            trigram_model=trigram_model,
            dictionary_set=dictionary_set,
        )

        print("\nRésultats GA :")
        print_summary([result_ga], "GA")
        print_best_runs([result_ga], "GA")

    elif args.algorithm == "eda":
        result_eda = run_eda_simple(
            seed=args.seed,
            nb_generations=args.nb_generations,
            nombre_prts=args.nombre_parents,
            population_size=args.population_size,
            perturbation_aleatoire=args.perturbation_rate,
            trigram_model=trigram_model,
            dictionary_set=dictionary_set,
            choice_indiv=args.choice_indiv
        )

        print("\nRésultats EDA :")
        print_summary([result_eda], "EDA")
        print_best_runs([result_eda], "EDA")

    else:
        menu()


if __name__ == "__main__":
    main()

