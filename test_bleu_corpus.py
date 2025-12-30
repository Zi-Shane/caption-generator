from nltk.translate.bleu_score import corpus_bleu

# Example data
# 2 samples
# Sample 1:
# Reference 1: "this is a test"
# Reference 2: "this is test"
# Hypothesis: "this is a test"

# Sample 2:
# Reference 1: "the quick brown fox"
# Hypothesis: "the quick brown fox jumped"

references = [
    [
        ['Two', 'teams', 'of', 'men', 'are', 'playing', 'soccer', 'on', 'a', 'grass', 'field'],
        ['A', 'man', 'is', 'playing', 'soccer'],
        ['Men', 'are', 'playing', 'a', 'football'],
        ['A', 'soccer', 'player', 'dribbles', 'past', 'his', 'opponents'],
        ['A', 'man', 'is', 'kicking', 'a', 'soccer', 'ball'],
        ['Peoples', 'are', 'playing', 'football'],
        ['Men', 'are', 'playing', 'soccer', 'on', 'a', 'field'],
        ['A', 'man', 'moves', 'a', 'soccer', 'ball', 'down', 'the', 'field'],
        ['Some', 'men', 'are', 'playing', 'soccer'],
        ['Men', 'are', 'playing', 'soccer'],
        ['People', 'are', 'playing', 'soccer'],
        ['A', 'man', 'is', 'kicking', 'a', 'soccer', 'ball'],
        ['The', 'man', 'tricked', 'the', 'soccer', 'player', 'in', 'dribbling', 'down', 'the', 'field'],
        ['The', 'men', 'are', 'playing', 'soccer'],
        ['A', 'fat', 'football', 'player', 'eludes', 'leaner', 'players'],
        ['A', 'man', 'dribbles', 'a', 'soccer', 'ball', 'down', 'the', 'field'],
        ['The', 'man', 'dribbled', 'the', 'soccer', 'ball', 'down', 'the', 'field']
    ],
    # [
    #     ['A', 'woman', 'is', 'placing', 'skewers', 'of', 'meat', 'on', 'some', 'kind', 'of', 'cooking', 'apparatus', '.'],
    #     ['A', 'person', 'sets', 'kebabs', 'on', 'a', 'grill', '.'],
    #     ['A', 'person', 'puts', 'skewered', 'meat', 'on', 'a', 'grill', '.'],
    #     ['A', 'woman', 'is', 'placing', 'skewers', 'of', 'chicken', 'over', 'a', 'grill', '.'],
    #     ['A', 'woman', 'is', 'putting', 'shish', 'kabobs', 'in', 'a', 'pan', 'to', 'cook', '.'],
    #     ['A', 'woman', 'is', 'frying', 'something', '.'],
    #     ['A', 'woman', 'is', 'placing', 'skewers', 'of', 'chicken', 'on', 'cooking', 'rack', '.'],
    #     ['A', 'person', 'is', 'placing', 'skewered', 'chicken', 'on', 'a', 'skewer', 'holder', '.'],
    #     ['A', 'woman', 'is', 'placing', 'skewered', 'food', 'onto', 'a', 'cooker', '.'],
    #     ['Skewers', 'of', 'meat', 'are', 'being', 'placed', 'on', 'a', 'grill', '.'],
    #     ['A', 'woman', 'is', 'placing', 'shish', 'kabobs', 'on', 'a', 'grill', '.'],
    #     ['The', 'woman', 'is', 'grilling', 'chicken', 'skewers', '.'],
    #     ['A', 'woman', 'is', 'placing', 'skewers', 'onto', 'a', 'rack', '.'],
    #     ['A', 'person', 'puts', 'skewered', 'meat', 'on', 'a', 'grill', '.'],
    #     ['The', 'lady', 'lined', 'up', 'the', 'shish-kabobs', 'on', 'the', 'tray', '.']
    # ],
    # [
    #     ['this', 'is', 'a', 'test'],
    #     ['this', 'is', 'test']
    # ],
    # [
    #     ['the', 'quick', 'brown', 'fox']
    # ]
]

hypotheses = [
    # ['Two', 'teams', 'of', 'people', 'are', 'playing', 'on', 'the', 'grass'],
    # ['A', 'woman', 'is', 'placing', 'skewers', 'of', 'meat', 'on', 'some', 'kind', 'of', 'cooking', 'apparatus', '.'],
    ['this', 'is', 'a', 'test'],
    # ['the', 'quick', 'brown', 'fox', 'jumped']
]

# Calculate BLEU-1
score_1 = corpus_bleu(references, hypotheses, weights=(1.0,))
print(f"BLEU-1: {score_1:.4f}")

# Calculate BLEU-4
score_4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
print(f"BLEU-4: {score_4:.4f}")

# Calculate default BLEU (BLEU-4)
score_default = corpus_bleu(references, hypotheses)
print(f"Default BLEU: {score_default:.4f}")
