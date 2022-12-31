import nltk


def tokenize(text):
    """Tokenize a string of text into words and punctuation."""
    tokens = nltk.word_tokenize(text)
    return tokens


def stem(tokens):
    """Stem a list of tokens using the Porter stemmer."""
    stemmer = nltk.PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens


def lemmatize(tokens):
    """Lemmatize a list of tokens using WordNet lemmatizer."""
    lemmatizer = nltk.WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens


def pos_tag(tokens):
    """Tag a list of tokens with part-of-speech tags."""
    tagged_tokens = nltk.pos_tag(tokens)
    return tagged_tokens


def ner(tokens):
    """Identify named entities in a list of tokens."""
    tagged_tokens = nltk.pos_tag(tokens)
    chunked_tokens = nltk.ne_chunk(tagged_tokens)
    named_entities = []
    for chunk in chunked_tokens:
        if isinstance(chunk, nltk.tree.Tree):
            label = chunk.label()
            entity_text = " ".join([token[0] for token in chunk])
            named_entities.append((label, entity_text))
    return named_entities


def chunk(tokens):
    """Chunk a list of tokens into noun phrases and verb phrases."""
    tagged_tokens = nltk.pos_tag(tokens)
    grammar = r"""
    NP: {<DT|PP\$>?<JJ>*<NN>}
    VP: {<VB.*>}
  """
    chunker = nltk.RegexpParser(grammar)
    chunked_tokens = chunker.parse(tagged_tokens)
    return chunked_tokens
