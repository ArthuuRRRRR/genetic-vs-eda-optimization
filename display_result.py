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


def plot_convergence(results_1, results_2, label_1="Méthode 1", label_2="Méthode 2"):
    generations = range(len(results_1[0]["history"]))

    median_1, q1_1, q3_1 = compute_median_q1_q3_best_curve(results_1)
    median_2, q1_2, q3_2 = compute_median_q1_q3_best_curve(results_2)

    plt.figure(figsize=(12, 6))

    plt.plot(generations, median_1, label=label_1, linewidth=2)
    plt.fill_between(generations, q1_1, q3_1, alpha=0.2)

    plt.plot(generations, median_2, label=label_2, linewidth=2)
    plt.fill_between(generations, q1_2, q3_2, alpha=0.2)

    plt.yscale("log")
    plt.xlabel("Génération")
    plt.ylabel("Meilleur score")
    plt.title("Convergence des scores")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_average_population(results_1, results_2, label_1="Méthode 1", label_2="Méthode 2"):
    avg_1 = compute_mean_avg_curve(results_1)
    avg_2 = compute_mean_avg_curve(results_2)

    generations = range(len(avg_1))

    plt.figure(figsize=(9, 5))
    plt.plot(generations, avg_1, linewidth=2, label=label_1)
    plt.plot(generations, avg_2, linewidth=2, label=label_2)

    plt.yscale("log")
    plt.xlabel("Génération")
    plt.ylabel("Score moyen")
    plt.title("Évolution moyenne de la population")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_boxplot(results_1, results_2, label_1="Méthode 1", label_2="Méthode 2"):
    final_1 = [run["best_score"] for run in results_1]
    final_2 = [run["best_score"] for run in results_2]

    plt.figure(figsize=(7, 5))
    plt.boxplot(
        [final_1, final_2],
        labels=[label_1, label_2],
        patch_artist=True
    )

    plt.ylabel("Meilleur score final")
    plt.title("Distribution des scores finaux")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_sorted_final_scores(results_1, results_2, label_1="Méthode 1", label_2="Méthode 2"):
    final_1 = sorted([run["best_score"] for run in results_1])
    final_2 = sorted([run["best_score"] for run in results_2])

    plt.figure(figsize=(9, 5))
    plt.plot(final_1, marker="o", linewidth=1.5, label=label_1)
    plt.plot(final_2, marker="o", linewidth=1.5, label=label_2)

    plt.xlabel("Run trié")
    plt.ylabel("Meilleur score final")
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