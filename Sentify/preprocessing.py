import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re, emoji, string

def preprocess_text(text):
    def replace_slang(text):
        words = text.split()
        replaced_words = []
        for word in words:
            synsets = wordnet.synsets(word)
            if synsets:
                replaced_words.append(synsets[0].lemmas()[0].name())
            else:
                replaced_words.append(word)
        return ' '.join(replaced_words)
    
    # Handle hashtags
    text = re.sub(r'#\w+', '', text)
    
    # Handle emojis
    text = emoji.demojize(text)
    
    # Replace slang words
    text = replace_slang(text)
    
    # Tokenization
    tokens = word_tokenize(text)

    # Lowercasing
    tokens = [token.lower() for token in tokens]
    
    # Negation handling
    negation_words = set(['not', 'no', 'never', 'none', 'neither', 'nor', 'cannot'])
    negated = False
    negation_scope = False

    for i in range(len(tokens)):
        if tokens[i] in negation_words:
            negated = True
            negation_scope = True
        elif negation_scope:
            if tokens[i] in string.punctuation:
                negation_scope = False
            else:
                tokens[i] += "_NEG"
        if negated and tokens[i] in string.punctuation:
            negated = False

    # Remove punctuation
    punctuation_list = string.punctuation
    tokens = [word for word in tokens if word not in punctuation_list]

    # POS tagging
    pos_tags = nltk.pos_tag(tokens)

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = []
    lemmatize_exceptions = set(['not', 'no', 'never', 'none', 'neither', 'nor', 'cannot'])  # Add more exceptions as needed

    for word, pos in pos_tags:
        if word not in lemmatize_exceptions:
            if pos.startswith('N'):  # N for Noun
                lemmatized_tokens.append(lemmatizer.lemmatize(word, pos='n'))
            elif pos.startswith('V'):  # V for Verb
                lemmatized_tokens.append(lemmatizer.lemmatize(word, pos='v'))
            elif pos.startswith('J'):   # J for Adjective
                lemmatized_tokens.append(lemmatizer.lemmatize(word, pos='a'))
            elif pos.startswith('R'):   # R for Adverb
                lemmatized_tokens.append(lemmatizer.lemmatize(word, pos='r'))
            else:
                lemmatized_tokens.append(word)
        else:
            lemmatized_tokens.append(word)

    lemmatized_text = ' '.join(lemmatized_tokens)

    return lemmatized_text
