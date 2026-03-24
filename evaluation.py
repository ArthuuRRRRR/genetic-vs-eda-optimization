from gen_lm import perplexité

def fonction_objective(trigram_model, word, dictionary, limite_size =16, minimum_size=4):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    penalty = 0.0
    if word in dictionary:
        penalty = penalty + 10000
    
    if len(word) > limite_size:
        penalty = penalty + 10000
    
    if len(word) < minimum_size:
        penalty = penalty + 10000
    
    if repetition_word(word):
        penalty = penalty + 10000
    
    if any(char not in alphabet for char in word):
        penalty = penalty + 10000
    
    return perplexité(word, trigram_model) + penalty



def repetition_word(word, max_consecutive=3):
    count = 1
    for i in range(1, len(word)):
        if word[i] == word[i-1]:
            count += 1
            if count > max_consecutive:
                return True
        else:
            count = 1
    return False

