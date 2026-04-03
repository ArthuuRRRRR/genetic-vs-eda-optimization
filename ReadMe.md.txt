
Par défaut, si aucun argument n est fourni en ligne de commande, j utilise un menu interactif
qui permet de lancer différentes analyses (comparaison GA vs EDA, analyse détaillée, génération d annexes, etc...).
Ce mode est pratique pour explorer rapidement les résultats sans avoir à retenir les paramètres.

J ai aussi ajouté un second mode basé sur argparse qui permet de lancer directement
un algorithme (GA ou EDA) depuis la ligne de commande en spécifiant les paramètres.
Cela permet de modifier les paramètres sans toucher au code. 

les commandes sont : 
Pour le menu et comparaison :
python main.py

Sinon juste pour run sans comparaison:
python main.py --algorithm ga --crossover_type uniform --etalon --elitisme 2 --losers 0 --reseed 2 --population_size 60 --nb_generations 40 --mutation_rate 0.2
python main.py --algorithm eda --population_size 60 --nb_generations 40 --perturbation_rate 0.2 --nombre_parents 20 --choice_indiv 2