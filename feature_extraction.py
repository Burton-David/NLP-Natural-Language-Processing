from collections import Counter
import math
import nltk


def word_count(text):
    """Count the number of words in a string of text."""
    tokens = nltk.word_tokenize(text)
    return len(tokens)


def unique_word_count(text):
    """Count the number of unique words in a string of text."""
    tokens = nltk.word_tokenize(text)
    unique_tokens = set(tokens)
    return len(unique_tokens)


def word_frequencies(text):
    """Compute the frequency of each word in a string of text."""
    tokens = nltk.word_tokenize(text)
    frequencies = nltk.FreqDist(tokens)
    return frequencies


def bigrams(text):
    """Generate bigrams from a string of text."""
    tokens = nltk.word_tokenize(text)
    bigrams = list(nltk.bigrams(tokens))
    return bigrams


def trigrams(text):
    """Generate trigrams from a string of text."""
    tokens = nltk.word_tokenize(text)
    trigrams = list(nltk.trigrams(tokens))
    return trigrams


def ngrams(text, n):
    """Generate n-grams from a string of text."""
    tokens = nltk.word_tokenize(text)
    ngrams = list(nltk.ngrams(tokens, n))
    return ngrams


def character_count(text):
    """Count the number of characters in a string of text."""
    return len(text)


def uppercase_count(text):
    """Count the number of uppercase characters in a string of text."""
    return sum(1 for c in text if c.isupper())


def lowercase_count(text):
    """Count the number of lowercase characters in a string of text."""
    return sum(1 for c in text if c.islower())


def digit_count(text):
    """Count the number of digits in a string of text."""
    return sum(1 for c in text if c.isdigit())


def punctuation_count(text):
    """Count the number of punctuation characters in a string of text."""
    return sum(1 for c in text if c in string.punctuation)


def whitespace_count(text):
    """Count the number of whitespace characters in a string of text."""
    return sum(1 for c in text if c.isspace())


def avg_word_length(text):
    """Compute the average length of the words in a string of text."""
    tokens = nltk.word_tokenize(text)
    avg_length = sum(len(token) for token in tokens) / len(tokens)
    return avg_length


def stopword_count(text):
    """Count the number of stopwords in a string of text."""
    stopwords = nltk.corpus.stopwords.words("english")
    tokens = nltk.word_tokenize(text)
    count = sum(1 for token in tokens if token.lower() in stopwords)
    return count


def sentence_count(text):
    """Count the number of sentences in a string of text."""
    sentences = nltk.sent_tokenize(text)
    return len(sentences)


def avg_sentence_length(text):
    """Compute the average length of the sentences in a string of text."""
    sentences = nltk.sent_tokenize(text)
    word_counts = [len(nltk.word_tokenize(sentence)) for sentence in sentences]
    avg_length = sum(word_counts) / len(word_counts)
    return avg_length


def flesch_reading_ease(text):
    """Compute the Flesch reading ease score for a string of text."""
    sentences = nltk.sent_tokenize(text)
    word_count = len(nltk.word_tokenize(text))
    syllable_count = sum(nltk.syllable_count(word)
                         for word in nltk.word_tokenize(text))
    score = 206.835 - 1.015 * (word_count / len(sentences)) - \
        84.6 * (syllable_count / word_count)
    return score


def flesch_kincaid_grade_level(text):
    """Compute the Flesch-Kincaid grade level for a string of text."""
    sentences = nltk.sent_tokenize(text)
    word_count = len(nltk.word_tokenize(text))
    syllable_count = sum(nltk.syllable_count(word)
                         for word in nltk.word_tokenize(text))
    score = 0.39


# TF-IDF


def tf(word, document):
  """Calculates the term frequency (TF) of a word in a document.
  TF is the number of times the word appears in the document, divided by the total number of words in the document."""
  return document.count(word) / len(document)


def idf(word, documents):
  """Calculates the inverse document frequency (IDF) of a word in a collection of documents.
  IDF is the logarithm of the total number of documents divided by the number of documents that contain the word."""
  num_documents = len(documents)
  num_documents_with_word = sum(
      1 for document in documents if word in document)
  return math.log(num_documents / num_documents_with_word)


def tfidf(word, document, documents):
  """Calculates the TF-IDF value of a word in a document in a collection of documents.
  TF-IDF is the product of the word's TF and IDF values."""
  return tf(word, document) * idf(word, documents)


def top_tfidf_words(document, documents, n=10):
  """Returns the top n words with the highest TF-IDF values in a document in a collection of documents."""
  words = Counter()
  for word in document:
    words[word] = tfidf(word, document, documents)
  return words.most_common(n)
