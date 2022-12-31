# NLP-Natural-Language-Processing
Commonly Used Code and Examples of NLP projects

## data_loading
* load_text_file(filename): Loads a text file and returns the contents as a string.
* load_csv_file(filename): Loads a CSV file and returns the contents as a Pandas DataFrame.
* load_webpage(url): Loads a webpage and returns the contents as an HTML string.
* parse_html(html): Parses an HTML string and returns a list of text nodes.
* preprocess_text(text): Preprocesses a string of text by lowercasing it and removing punctuation.

## Preprocess
* tokenize(text): Tokenizes a string of text into words and punctuation.
* stem(tokens): Stems a list of tokens using the Porter stemmer.
* lemmatize(tokens): Lemmatizes a list of tokens using the WordNet lemmatizer.
* pos_tag(tokens): Tags a list of tokens with part-of-speech tags.
* ner(tokens): Identifies named entities in a list of tokens.
* chunk(tokens): Chunks a list of tokens into noun phrases and verb phrases.

## Feature_extraction
* word_count(text): Counts the number of words in a string of text.
* unique_word_count(text): Counts the number of unique words in a string of text.
* word_frequencies(text): Computes the frequency of each word in a string of text.
* bigrams(text): Generates bigrams (sequences of two consecutive tokens) from a string of text.
* trigrams(text): Generates trigrams (sequences of three consecutive tokens) from a string of text.
* ngrams(text, n): Generates n-grams (sequences of n consecutive tokens) from a string of text.
* character_count(text): Counts the number of characters (letters, digits, punctuation, etc.) in a string of text.
* uppercase_count(text): Counts the number of uppercase characters in a string of text.
* lowercase_count(text): Counts the number of lowercase characters in a string of text.
* digit_count(text): Counts the number of digits in a string of text.
* punctuation_count(text): Counts the number of punctuation characters in a string of text.
* whitespace_count(text): Counts the number of whitespace characters (spaces, tabs, newlines) in a string of text.
* avg_word_length(text): Computes the average length (in characters) of the words in a string of text.
* stopword_count(text): Counts the number of stopwords (common, unimportant words) in a string of text.
* sentence_count(text): Counts the number of sentences in a string of text.
* avg_sentence_length(text): Computes the average length (in words) of the sentences in a string of text.
* flesch_reading_ease(text): Computes the Flesch reading ease score for a string of text.
* flesch_kincaid_grade_level(text): Computes the Flesch-Kincaid grade level for a string of text.
* tf(word, document): Calculates the term frequency (TF) of a word in a document. TF is the number of times the word appears in the document, divided by the total number of words in the document.
* idf(word, documents): Calculates the inverse document frequency (IDF) of a word in a collection of documents. IDF is the logarithm of the total number of documents divided by the number of documents that contain the word.
* tfidf(word, document, documents): Calculates the TF-IDF value of a word in a document in a collection of documents. TF-IDF is the product of the word's TF and IDF values.
* top_tfidf_words(document, documents, n=10): Returns the top n words with the highest TF-IDF values in a document in a collection of documents.

## machine-learning
* train_classifier(X, y, model_type): Trains a classifier model on a training dataset X and labels y. model_type specifies the type of model to use (e.g., "logistic regression", "support vector machine", "neural network"). Returns the trained model.
* evaluate_classifier(X, y, model): Evaluates a classifier model on a test dataset X and labels y. Returns the model's accuracy score.
predict_class(X, model): Makes class predictions using a classifier model on a new dataset X. Returns the predictions.
* train_sentiment_analyzer(X, y, model_type): Trains a sentiment analysis model on a training dataset X and labels y. model_type specifies the type of model to use (e.g., "logistic regression", "support vector machine", "neural network"). Returns the trained model.
* evaluate_sentiment_analyzer(X, y, model): Evaluates a sentiment analysis model on a test dataset X and labels y. Returns the model's accuracy score.
* predict_sentiment(X, model): Makes sentiment predictions using a sentiment analysis model on a new dataset X. Returns the predictions.
* train_translator(X, y, model_type): Trains a machine translation model on a training dataset X and labels y. model_type specifies the type of model to use (e.g., "sequence-to-sequence", "transformer"). Returns the trained model.
* evaluate_translator(X, y, model): Evaluates a machine translation model on a test dataset X and labels y. Returns the model's accuracy score.
* translate(X, model): Translates a string of text X using a machine translation model model. Returns the translated string.

## evaluation
* accuracy: Calculates the accuracy of a model's predictions. Accuracy is the number of correct predictions divided by the total number of predictions.
* precision: Calculates the precision of a model's predictions. Precision is the number of true positive predictions divided by the total number of positive predictions.
* recall: Calculates the recall of a model's predictions. Recall is the number of true positive predictions divided by the total number of actual positive instances.
* f1_score: Calculates the F1 score of a model's predictions. F1 score is the harmonic mean of precision and recall.
* confusion_matrix: Calculates the confusion matrix for a model's predictions. The confusion matrix is a table of the true positive, true negative, false positive, and false negative predictions for each label.

## data_pipeline
* load_tsv: Loads a TSV file and returns a pandas DataFrame.
* load_csv: Loads a CSV file and returns a pandas DataFrame.
* load_excel: Loads an Excel file and returns a pandas DataFrame.
* preprocess_text: Applies a text preprocessor function to a specified text column in a DataFrame and returns a new DataFrame with the preprocessed text column.
* split_data: Splits a DataFrame into training and testing sets and returns a tuple of (X_train, X_test, y_train, y_test) arrays.