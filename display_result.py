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
    generations = range(len(results_ga[0]["history"]))

    median_ga, q1_ga, q3_ga = compute_median_q1_q3_best_curve(results_ga)
    median_eda, q1_eda, q3_eda = compute_median_q1_q3_best_curve(results_eda)

    plt.figure(figsize=(12, 6))  

    plt.plot(generations, median_ga, label="GA", linewidth=2)
    plt.fill_between(generations, q1_ga, q3_ga, alpha=0.2)

    plt.plot(generations, median_eda, label="EDA", linewidth=2)
    plt.fill_between(generations, q1_eda, q3_eda, alpha=0.2)

    plt.yscale("log")

    plt.xlabel("Génération")
    plt.ylabel("Meilleur score")
    plt.title("Convergence des scores")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def compute_median_q1_q3_best_curve(results):
    nb_generations = len(results[0]["history"])
    median_curve = []
    q1_curve = []
    q3_curve = []

    for g in range(nb_generations):
        scores = [run["history"][g]["best_score"] for run in results]
        median_curve.append(np.median(scores))
        q1_curve.append(np.percentile(scores, 25))
        q3_curve.append(np.percentile(scores, 75))

    return median_curve, q1_curve, q3_curve



def plot_average_population(results_ga, results_eda):
    avg_ga = compute_mean_avg_curve(results_ga)
    avg_eda = compute_mean_avg_curve(results_eda)

    generations = range(len(avg_ga))

    plt.figure(figsize=(9, 5))
    plt.plot(generations, avg_ga, linewidth=2, label="GA")
    plt.plot(generations, avg_eda, linewidth=2, label="EDA")

    plt.yscale("log")
    plt.xlabel("Génération")
    plt.ylabel("Score moyen")
    plt.title("Évolution moyenne de la population")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_boxplot(results_ga, results_eda):
    ga_final = [run["best_score"] for run in results_ga]
    eda_final = [run["best_score"] for run in results_eda]

    plt.figure(figsize=(7, 5))
    plt.boxplot(
        [ga_final, eda_final],
        labels=["GA", "EDA"],
        patch_artist=True
    )

    plt.ylabel("Meilleur score final")
    plt.title("Distribution des scores finaux")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_sorted_final_scores(results_ga, results_eda):
    ga_final = sorted([run["best_score"] for run in results_ga])
    eda_final = sorted([run["best_score"] for run in results_eda])

    plt.figure(figsize=(9, 5))
    plt.plot(ga_final, marker="o", linewidth=1.5, label="GA")
    plt.plot(eda_final, marker="o", linewidth=1.5, label="EDA")

    plt.xlabel("Run trié")
    plt.ylabel("Best score final")
    plt.title("Scores finaux triés par run")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def print_summary(results, algo_name="GA"):
    final_scores = [run["best_score"] for run in results]

    print(f"\n--- Résumé {algo_name} ---")
    print("Nombre de runs :", len(results))
    print("Moyenne:", np.mean(final_scores))
    print("Écart-type:", np.std(final_scores))
    print("Minimum:", np.min(final_scores))
    print("Médiane:", np.median(final_scores))
    print("Maximum:", np.max(final_scores))


def print_best_runs(results, algo_name="GA", top_n=3):
    sorted_results = sorted(results, key=lambda x: x["best_score"])

    print(f"\n--- Meilleurs runs {algo_name} ---")
    for i, run in enumerate(sorted_results[:top_n], start=1):
        print(
            f"{i}. seed={run['seed']} | "
            f"best_word={run['best_word']} | "
            f"best_score={run['best_score']}")

