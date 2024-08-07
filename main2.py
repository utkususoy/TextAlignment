import re
from fuzzywuzzy import process
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# Preprocess the text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text


# Spell correction using fuzzy matching
def correct_spelling(word, vocabulary):
    #TODO score_cutoff parametresini incele

    best_match, score = process.extractOne(word, vocabulary)
    return best_match[0]


# Load pre-trained fastText vectors (ensure you have the model downloaded)
def load_model():
    return KeyedVectors.load_word2vec_format('./cc.tr.300.vec', binary=False)

# Get word vector from the model
def get_word_vector(model, word):
    if word in model:
        return model[word]
    else:
        return None


# Compute cosine similarity
def cosine_similarity_score(vec1, vec2):
    return cosine_similarity([vec1], [vec2])[0][0]


# Get similarity scores for each word in the sentence based on the target word
def get_similarity_scores(sentence, target_word, model):
    sentence = preprocess_text(sentence)
    words = sentence.split()
    target_word = correct_spelling(target_word, words)

    target_vector = get_word_vector(model, target_word)
    if target_vector is None:
        return None

    scores = []
    for word in words:
        word_vector = get_word_vector(model, word)
        if word_vector is not None:
            score = cosine_similarity_score(target_vector, word_vector)
            scores.append((word, score))
        else:
            scores.append((word, 0))  # No vector found

    return scores


# Main execution
if __name__ == "__main__":
    sentence = "Bu bir örnek cümledir."
    target_word = "örnek"  # Misspelled word for 'örnek'

    # Load the pre-trained model
    # model = load_model()

    # Get similarity scores
    scores = get_similarity_scores(sentence, target_word, None)
    if scores:
        for word, score in scores:
            print(f"Word: {word}, Similarity Score: {score:.4f}")
    else:
        print("Target word vector not found in the model.")
