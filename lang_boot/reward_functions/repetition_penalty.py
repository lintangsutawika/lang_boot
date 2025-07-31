def repetition_penalty(text, window_size=5, penalty_factor=1.0):
    """
    Calculate a penalty score for repetitive text based on n-gram frequency.
    
    Args:
        text (str): Input text to analyze
        window_size (int): Size of n-grams to check (default: 5 for 5-grams)
        penalty_factor (float): Multiplier for penalty calculation (default: 1.0)
    
    Returns:
        float: Penalty score (higher = more repetitive)
    """
    if len(text) < window_size:
        return 0.0
    
    # Convert to lowercase and split into tokens
    tokens = text.lower().split()
    
    if len(tokens) < window_size:
        return 0.0
    
    # Create n-grams
    ngrams = []
    for i in range(len(tokens) - window_size + 1):
        ngram = ' '.join(tokens[i:i + window_size])
        ngrams.append(ngram)
    
    # Count frequency of each n-gram
    ngram_counts = {}
    for ngram in ngrams:
        ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
    
    # Calculate penalty based on repetitions
    penalty = 0.0
    total_ngrams = len(ngrams)
    
    for count in ngram_counts.values():
        if count > 1:
            # Penalty increases exponentially with repetition
            penalty += (count - 1) ** 2 * penalty_factor
    
    # Normalize by total number of n-grams
    normalized_penalty = penalty / total_ngrams if total_ngrams > 0 else 0.0
    
    return normalized_penalty


def word_repetition_penalty(text, penalty_factor=1.0):
    """
    Simpler version that penalizes repeated words.
    
    Args:
        text (str): Input text to analyze
        penalty_factor (float): Multiplier for penalty calculation
    
    Returns:
        float: Penalty score based on word repetition
    """
    words = text.lower().split()
    
    if len(words) == 0:
        return 0.0
    
    # Count word frequencies
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    # Calculate penalty
    penalty = 0.0
    for count in word_counts.values():
        if count > 1:
            penalty += (count - 1) * penalty_factor
    
    # Normalize by total words
    return penalty / len(words)


def character_repetition_penalty(text, min_length=3):
    """
    Penalize repeated character sequences.
    
    Args:
        text (str): Input text to analyze
        min_length (int): Minimum length of repeated sequences to penalize
    
    Returns:
        float: Penalty score for character repetition
    """
    if len(text) < min_length * 2:
        return 0.0
    
    penalty = 0.0
    text_lower = text.lower()
    
    # Check for repeated substrings
    for length in range(min_length, len(text) // 2 + 1):
        for i in range(len(text) - length + 1):
            substring = text_lower[i:i + length]
            
            # Count occurrences of this substring
            count = 0
            start = 0
            while True:
                pos = text_lower.find(substring, start)
                if pos == -1:
                    break
                count += 1
                start = pos + 1
            
            if count > 1:
                # Longer repeated sequences get higher penalty
                penalty += (count - 1) * length * 0.1
    
    return penalty / len(text)


# Example usage
if __name__ == "__main__":
    # Test cases
    test_texts = [
        "Untuk mengetahuinya, bisa hitung jumlahnya.\n\n1. Hitung jumlahnya dari jumlahnya. Jadi, jumlahnya dari jumlahnya menjadi 16 - 3 - 4 = 9.\n\n2. Jadi, jumlahnya jadi 9.\n\n3. Untuk hitungnya, jadi bisa jadi 9 × 2 = 18.\n\nJadi, jumlahnya dari jumlahnya menjadi 18. Jadi, hasilnya menjadi 18. \n\nJadi, jumlahnya dari hasilnya menjadi 18. Jadi, jawabannya menjadi 18.",
        "This is a normal sentence with no repetition.",
        "The cat sat on the mat. The cat sat on the mat.",
        "Hello hello hello world world world",
        "I love programming. Programming is fun. I really love programming.",
        "aaaaaaaaaa bbbbbbbbbb cccccccccc"
    ]
    
    print("N-gram Repetition Penalty:")
    for text in test_texts:
        penalty = 0
        for N in [1, 2, 3, 4]:
            N_penalty = repetition_penalty(text, window_size=N)
            print(f"N={N}: '{text}' -> Penalty: {N_penalty:.3f}")
            penalty += N_penalty
        penalty /= 3
        print(f"'{text}' -> Penalty: {penalty:.3f}")
    
    # print("\nWord Repetition Penalty:")
    # for text in test_texts:
    #     penalty = word_repetition_penalty(text)
    #     print(f"'{text}' -> Penalty: {penalty:.3f}")
    
    # print("\nCharacter Repetition Penalty:")
    # for text in test_texts:
    #     penalty = character_repetition_penalty(text)
    #     print(f"'{text}' -> Penalty: {penalty:.3f}")
