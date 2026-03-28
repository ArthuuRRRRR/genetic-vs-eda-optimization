from ga import ga
from eda import eda

import random
import numpy as np


def run_ga_simple(seed,nb_generations,population_size,mutation_rate_pm,trigram_model,dictionary_set,choice_indiv=2,crossover_type="one_point",elitisme=2, etalon=False, losers=0, reseed=2):
    random.seed(seed)
    np.random.seed(seed)

    ga_model = ga(population_size=population_size,mutation_rate_pm=mutation_rate_pm,trigram_model=trigram_model,dictionary_set=dictionary_set,choice_indiv=choice_indiv,crossover_type=crossover_type,elitisme=elitisme, etalon=etalon, losers=losers, reseed=seed)

    result = ga_model.run(nb_generations)

    best_word, best_score, history = result

    return {"seed": seed,"best_word": best_word,"best_score": best_score,"history": history}


def run_eda_simple(seed,nb_generations,nombre_prts,population_size,mutation_rate, trigram_model,dictionary_set,choice_indiv):
    random.seed(seed)
    np.random.seed(seed)

    eda_model = eda(population_size=population_size,mutation_rate=mutation_rate,trigram_model=trigram_model,dictionary_set=dictionary_set,choice_indiv=choice_indiv)

    result = eda_model.run(nb_generations, nombre_prts)

    best_word, best_score, history = result

    return {"seed": seed,"best_word": best_word,"best_score": best_score,"history": history}



def monte_carlo_ga(n_runs,nb_generations,population_size,mutation_rate_pm,trigram_model,dictionary_set,choice_indiv=2,crossover_type="one_point", elitisme=2, etalon=False, losers=0, reseed=2):
    all_results = []

    for seed in range(n_runs):
        result = run_ga_simple(seed=seed,nb_generations=nb_generations,population_size=population_size,mutation_rate_pm=mutation_rate_pm,trigram_model=trigram_model,dictionary_set=dictionary_set,choice_indiv=choice_indiv,crossover_type=crossover_type,elitisme=elitisme, etalon=etalon, losers=losers, reseed=reseed)
        all_results.append(result)

    return all_results


def monte_carlo_eda(n_runs,nb_generations,nombre_prts,population_size,mutation_rate,trigram_model,dictionary_set, choice_indiv):
    all_results = []

    for seed in range(n_runs):
        result = run_eda_simple(seed=seed,nb_generations=nb_generations,nombre_prts=nombre_prts,population_size=population_size,mutation_rate=mutation_rate,trigram_model=trigram_model,dictionary_set=dictionary_set,choice_indiv=choice_indiv)
        all_results.append(result)

    return all_results