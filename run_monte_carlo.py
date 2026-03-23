from ga import ga
from eda import eda


import random
import numpy as np


def run_ga_simple(seed, n_generations, population_size, mutation_rate, trigram_model, dictionary_set):
    random.seed(seed)
    np.random.seed(seed)

    model = ga(population_size=population_size,mutation_rate=mutation_rate,trigram_model=trigram_model,dictionary_set=dictionary_set)

    history = []

    for generation in range(n_generations):
        best_word, best_score, avrg_score = model.run()

        history.append({"nbr_generation": generation,"best_score": best_score,"avrg_score": avrg_score})

    return history


def run_eda_simple(seed, n_generations, population_size, trigram_model, dictionary_set):
    random.seed(seed)
    np.random.seed(seed)

    model = eda(population_size=population_size,trigram_model=trigram_model,dictionary_set=dictionary_set)

    history = []

    for generation in range(n_generations):
        best_word, best_score, avrg_score = model.run()

        history.append({"nbr_generation": generation,"best_score": best_score,"avrg_score": avrg_score})

    return history

