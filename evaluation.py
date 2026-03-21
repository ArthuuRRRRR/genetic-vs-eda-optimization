from gen_lm import perplexité

def fonction_objective(trigram_model, word, dictionary, limite_size =16):
    penalty = 0.0
    if word in dictionary:
        penalty = penalty + 10000
    
    if len(word) > limite_size:
        penalty = penalty + 10000
    
    if len(word) <= 4:
        penalty = penalty + 10000
    
    return perplexité(word, trigram_model) + penalty