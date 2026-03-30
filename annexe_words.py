from evaluation import fonction_objective
import csv

def collect_words(results):
    words = set()

    for run in results:
        for gen in run["history"]:
            words.add(gen["best_word"])

    return words


def clean_words(words, dictionary_set):
    return [mot for mot in words if mot not in dictionary_set]


def rank_words(words, trigram_model, dictionary_set):
    position = []

    for w in words:
        score = fonction_objective(trigram_model, w, dictionary_set)
        position.append((w, score))

    position.sort(key=lambda x: x[1])
    return position


def generate_annexe(results, trigram_model, dictionary_set, filename):
    words = collect_words(results)
    words = clean_words(words, dictionary_set)
    position = rank_words(words, trigram_model, dictionary_set)

    top500_mots = position[:500]

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mot", "score"])

        for w, s in top500_mots:
            writer.writerow([w, s])

    print("c'est bon")