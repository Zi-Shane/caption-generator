
import json
import os
import sys

from code.vocabulary import Vocabulary

def verify_vocab():
    # Setup dummy data
    data = [
        {"caption": ["hello world"]},
        {"caption": ["hello python"]},
        {"caption": ["test test test"]},
        {"caption": ["rare word"]}
    ]
    
    # "hello": 2
    # "world": 1
    # "python": 1
    # "test": 3
    # "rare": 1
    # "word": 1
    
    test_file = "test_captions.json"
    with open(test_file, 'w') as f:
        json.dump(data, f)
        
    print("--- Test 1: min_count=1 (Default) ---")
    vocab = Vocabulary()
    vocab.build_vocab(test_file, min_count=1)
    
    expected_words = ["hello", "world", "python", "test", "rare", "word"]
    for w in expected_words:
        assert w in vocab.vocab, f"Expected '{w}' to be in vocab (min_count=1)"
    print("Passed.")

    print("\n--- Test 2: min_count=2 ---")
    vocab = Vocabulary()
    vocab.build_vocab(test_file, min_count=2)
    
    assert "hello" in vocab.vocab, "Expected 'hello' (count 2) in vocab"
    assert "test" in vocab.vocab, "Expected 'test' (count 3) in vocab"
    assert "world" not in vocab.vocab, "Expected 'world' (count 1) NOT in vocab"
    assert "python" not in vocab.vocab, "Expected 'python' (count 1) NOT in vocab"
    print("Passed.")

    print("\n--- Test 3: min_count=3 ---")
    vocab = Vocabulary()
    vocab.build_vocab(test_file, min_count=3)
    
    assert "test" in vocab.vocab, "Expected 'test' (count 3) in vocab"
    assert "hello" not in vocab.vocab, "Expected 'hello' (count 2) NOT in vocab"
    print("Passed.")
    
    # Cleanup
    if os.path.exists(test_file):
        os.remove(test_file)
        
    print("\nAll verification tests passed!")

if __name__ == "__main__":
    verify_vocab()
