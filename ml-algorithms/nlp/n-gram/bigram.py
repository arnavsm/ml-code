import re
from collections import defaultdict

# Sample text
# text = "The quick brown fox jumps over the lazy dog. The dog barks at the fox."
with open('more.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Preprocess the text
words = re.findall(r'\w+', text.lower())

# Create bigrams
bigrams = list(zip(words, words[1:]))

# Count bigram frequencies
bigram_counts = defaultdict(int)
for bigram in bigrams:
    bigram_counts[bigram] += 1

# Calculate probabilities
bigram_probabilities = {}
for bigram, count in bigram_counts.items():
    word1, word2 = bigram
    bigram_probabilities[bigram] = count / sum(1 for bg in bigrams if bg[0] == word1)

# Print some example probabilities
print("Bigram Probabilities:")
for bigram, prob in bigram_probabilities.items():
    print(f"{bigram}: {prob:.2f}")

# Predict the next word
def predict_next_word(word):
    candidates = [(w2, p) for (w1, w2), p in bigram_probabilities.items() if w1 == word]
    if candidates:
        return max(candidates, key=lambda x: x[1])[0]
    return "No prediction"

# Example predictions
print("\nPredictions:")
print(f"After 'the': {predict_next_word('the')}")
print(f"After 'he': {predict_next_word('he')}")