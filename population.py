import random
import string


def generate_random_word( min_length= 4, max_length=16):
    longueur = random.randint(min_length, max_length)
    return "".join(random.choice(string.ascii_lowercase) for _ in range(longueur))



def init_population(size=100, word_length=5):
    population = [generate_random_word(word_length) for _ in range(size)]
    return population
