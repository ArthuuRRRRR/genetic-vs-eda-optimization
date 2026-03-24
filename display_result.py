import numpy as np
import matplotlib.pyplot as plt


def compute_mean_best_curve(results):
    nb_generations = len(results[0]["history"])
    mean_curve = []

    for g in range(nb_generations):
        scores = [run["history"][g]["best_score"] for run in results]
        mean_curve.append(np.mean(scores))

    return mean_curve


def compute_mean_avg_curve(results):
    nb_generations = len(results[0]["history"])
    mean_curve = []

    for g in range(nb_generations):
        scores = [run["history"][g]["average_score"] for run in results]
        mean_curve.append(np.mean(scores))

    return mean_curve


def plot_convergence(results_ga, results_eda):
    mean_ga = compute_mean_best_curve(results_ga)
    mean_eda = compute_mean_best_curve(results_eda)

    plt.figure()
    plt.plot(mean_ga, label="GA")
    plt.plot(mean_eda, label="EDA")
    plt.yscale("log")

    plt.xlabel("Generation")
    plt.ylabel("Best score (mean)")
    plt.title("Convergence moyenne")
    plt.legend()
    plt.grid()

    plt.show()

def plot_average_population(results_ga, results_eda):
    avg_ga = compute_mean_avg_curve(results_ga)
    avg_eda = compute_mean_avg_curve(results_eda)

    plt.figure()
    plt.plot(avg_ga, label="GA avg")
    plt.plot(avg_eda, label="EDA avg")
    plt.yscale("log")

    plt.xlabel("Generation")
    plt.ylabel("Average score")
    plt.title("Évolution moyenne de la population")
    plt.legend()
    plt.grid()

    plt.show()

def plot_boxplot(results_ga, results_eda):
    ga_final = [run["best_score"] for run in results_ga]
    eda_final = [run["best_score"] for run in results_eda]

    plt.figure()
    plt.boxplot([ga_final, eda_final], labels=["GA", "EDA"])

    plt.ylabel("Best score final")
    plt.title("Distribution des scores finaux")

    plt.grid()
    plt.show()
