import numpy as np

# =====================================================
# 1. Vocabulary
# =====================================================

VOCAB = "abcdefghijklmnopqrstuvwxyz .'"

K = len(VOCAB)

char_to_idx = {
    ch: i
    for i, ch in enumerate(VOCAB)
}

idx_to_char = {
    i: ch
    for i, ch in enumerate(VOCAB)
}

print(f"Vocabulary size: {K}")

# =====================================================
# 2. Load Shakespeare Corpus
# =====================================================

with open("shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

# Keep only supported characters
text = ''.join(
    ch
    for ch in text
    if ch in VOCAB
)

print("Corpus length:", len(text))

# =====================================================
# 3. Create Statistics Table
# =====================================================

S = np.zeros(
    (K, K, K, K),
    dtype=np.int32
)

print("Training model...")

for i in range(len(text) - 3):

    a = char_to_idx[text[i]]
    b = char_to_idx[text[i + 1]]
    c = char_to_idx[text[i + 2]]
    d = char_to_idx[text[i + 3]]

    S[a, b, c, d] += 1

print("Training complete.")

# =====================================================
# 4. Conditional Probability
# =====================================================

def conditional_probability(a, b, c, d, S, K):

    numerator = S[a, b, c, d] + 1

    denominator = (
        np.sum(S[a, b, c, :])
        + K
    )

    return numerator / denominator

# =====================================================
# 5. Sentence Probability
# =====================================================

def sentence_probability(
        sentence,
        S,
        char_to_idx,
        K):

    sentence = sentence.lower()

    sentence = ''.join(
        ch
        for ch in sentence
        if ch in char_to_idx
    )

    if len(sentence) < 4:
        return 0.0

    probability = 1.0

    for i in range(3, len(sentence)):

        a = char_to_idx[sentence[i - 3]]
        b = char_to_idx[sentence[i - 2]]
        c = char_to_idx[sentence[i - 1]]
        d = char_to_idx[sentence[i]]

        probability *= conditional_probability(
            a,
            b,
            c,
            d,
            S,
            K
        )

    return probability

# =====================================================
# 6. Sentence Log Probability
# =====================================================

def sentence_log_probability(
        sentence,
        S,
        char_to_idx,
        K):

    sentence = sentence.lower()

    sentence = ''.join(
        ch
        for ch in sentence
        if ch in char_to_idx
    )

    if len(sentence) < 4:
        return float('-inf')

    log_probability = 0.0

    for i in range(3, len(sentence)):

        a = char_to_idx[sentence[i - 3]]
        b = char_to_idx[sentence[i - 2]]
        c = char_to_idx[sentence[i - 1]]
        d = char_to_idx[sentence[i]]

        p = conditional_probability(
            a,
            b,
            c,
            d,
            S,
            K
        )

        log_probability += np.log(p)

    return log_probability

# =====================================================
# 7. Average Log Probability
# =====================================================
#
# Useful for comparing texts of
# different lengths.
#
# =====================================================

def average_log_probability(
        sentence,
        S,
        char_to_idx,
        K):

    sentence = sentence.lower()

    sentence = ''.join(
        ch
        for ch in sentence
        if ch in char_to_idx
    )

    if len(sentence) < 4:
        return float('-inf')

    total_log_prob = 0.0
    count = 0

    for i in range(3, len(sentence)):

        a = char_to_idx[sentence[i - 3]]
        b = char_to_idx[sentence[i - 2]]
        c = char_to_idx[sentence[i - 1]]
        d = char_to_idx[sentence[i]]

        p = conditional_probability(
            a,
            b,
            c,
            d,
            S,
            K
        )

        total_log_prob += np.log(p)
        count += 1

    return total_log_prob / count

# =====================================================
# 8. Assignment Evaluation
# =====================================================

sentence1 = "to be or not to be"
sentence2 = "xfsdgfsf"

print("\n==============================")
print("Sentence 1:", sentence1)
print("==============================")

print(
    "Probability:",
    sentence_probability(
        sentence1,
        S,
        char_to_idx,
        K
    )
)

print(
    "Log Probability:",
    sentence_log_probability(
        sentence1,
        S,
        char_to_idx,
        K
    )
)

print(
    "Average Log Probability:",
    average_log_probability(
        sentence1,
        S,
        char_to_idx,
        K
    )
)

print("\n==============================")
print("Sentence 2:", sentence2)
print("==============================")

print(
    "Probability:",
    sentence_probability(
        sentence2,
        S,
        char_to_idx,
        K
    )
)

print(
    "Log Probability:",
    sentence_log_probability(
        sentence2,
        S,
        char_to_idx,
        K
    )
)

print(
    "Average Log Probability:",
    average_log_probability(
        sentence2,
        S,
        char_to_idx,
        K
    )
)