# Natural Language Processing

* A field of *Artificial Intelligence* that gives the machines the ability to read, understand and derive meaning from human languages


<br>


## <u>***Preprocessing Text (Unstructured data)***</u>

*Tokenization & Stop words*
* Tokenization is the process to break down documents into smaller units of analysis
* It's best practice to use different stop words for different corpus
* The more stopwords, the fewer words we have to look through in output but this increases the chances that we delete informative words

<br>

 *Lemmatization*
 * A technique that transforms various morphologies of a word into its base form
 * Takes words in different forms (past tense, plural, etc.) and transforms them into the base form (present tense, singular)
 * It is not applicable to any kind of analysis - measures of sentiment would be biased if the words were adjective-lemmatized beforehand


*Steps*
    
1. Regex substitution 
2. word tokenization
3. Lemmatizer
4. Stop Words

 ```
from nltk.corpus import reuters, stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

# Get all articles that are related to gas

ids = reuters.fileids(categories='gas')
corpus = [reuters.raw(i) for i in ids]


# Define a function to preprocess the article

def process_text(article):

    regex = re.compile("[^a-zA-Z ]")
    re_clean = regex.sub('', article)

    words = word_tokenize(re_clean)

    lem = [lemmatizer.lemmatize(word) for word in words]

    sw = set(stopwords.words('english')).union({'hey', 'hi'}
    output = [word.lower() for word in lem if word.lower() not in sw]

    return output
```


<br>

*Ngram Counter*
* Another way of tokenizing articles
* `N` stands for the number of consecutive words (or tokens) that are included

```
# get word counts

from collections import Counter
word_counts = Counter(processed)
print(dict(word_counts))

from nltk.util import ngrams
bigram_counts = Counter(ngrams(processed, n=2))
print(dict(bigram_counts.most_common(10)))


# Define the counter function

def word_counter(corpus): 

    # Combine all articles in corpus into one large string
    big_string = ' '.join(corpus)

    processed = process_text(big_string)
    word_counts = Counter(processed)
    top_10 = dict(word_counts.most_common(10))
    
    return pd.DataFrame(list(top_10.items()), columns=['word', 'count'])

word_counter(corpus)    
```


<br>

*Word Cloud*
* The word cloud function takes only a single string as an argument, we must join these words back together in the preprocessing function
```
from wordcloud import WordCloud

big_string = ' '.join(corpus)
input_words = process_text(big_string)
input_words_str = ' '.join(input_words)

wc = WordCloud(width=800, height=600, max_words=28).generate(input_words_str)
plt.imshow(wc)
```


<br>
<br>

## <u>***Sentiment Analysis***</u>

*Terms Relevance*
* A measure to understand how important a word is to a corpus, which is a large, structured, and organized collection of text documents that normally focuses on a specific matter


*TF–IDF*
* A weighting factor intended to measure how important a word is to a document in a corpus

* `TF` indicates that if a word appears multiple times in a document, it can be concluded that it is relevant and more meaningful than other words in the same text

* `IDF` comes to action when you are analyzing several documents. If a word also appears many times among a collection of documents, maybe it's just a frequent word and not a relevant one.


```
from sklearn.feature_extraction.text import TfidfVectorizer

# Get the first 50 corpus from Reuters dataset

all_docs_id = reuters.fileids()
corpus_id = all_docs_id[0:50]
corpus = [reuters.raw(doc).lower() for doc in corpus_id]

# Create TF-IDF instance
vectorizer = TfidfVectorizer(stop_words="english")
X_corpus = vectorizer.fit_transform(corpus)


# Output of X_corpus:
# (n, m)
# where n = number of documents
#       m = number of unique words/tokens



# Retrieve words list from corpus
words_corpus = vectorizer.get_feature_names()

# Put the TF-IDF weight of each word of words into dataframe

words_corpus_df = pd.DataFrame(
    list(zip(words_corpus, np.ravel(X_corpus.mean(axis=0)))), columns=["Word", "TF-IDF"]
)

words_corpus_df = words_corpus_df.sort_values(by=["TF-IDF"], ascending=False)

```

<br>



*Bag-of-Words*

* A technique to represent the important words/tokens in a document without worrying about sentence structure

```
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import reuters

doc_id = "test/14826"
doc_text = reuters.raw(doc_id)

# Create CountVectorizer instance
vectorizer = CountVectorizer(stop_words="english")

# Get the tokenization and occurrence counting
# X raw data contains the occurrence of each term in the document
# A unique ID is assigned to each term

X = vectorizer.fit_transform([doc_text])


# Output of X:
# (n,t) c
# where n = the nth document, 
#       t = the term's numeric identifier
#       c = number of occurrence

# Retrieve unique words list
words = vectorizer.get_feature_names()


# Put the bag of words into dataframe
words_df = pd.DataFrame(
    list(zip(words, np.ravel(X.sum(axis=0)))), 
    columns=["Word", "Word_Count"]
)
```

<br>

More Reuters methods
```
# Get totalnumber of articles
len(retrieve_docs(["US", "COVID"]))


# Get all the categories associated with a document
reuters.categories('training/14826')

# Get the tokenized version of each article
reuters.words('training/14826')

# Pass a list of terms as parameter and get all articles that contains the search terms

news_ids= ['test/123', 'test/456', 'test/789']

def retrieve_docs(search_terms):
    result_docs = []
    for doc_id in news_ids:
        found_terms = [
            word
            for word in reuters.words(doc_id)
            if any(term in word.lower() for term in search_terms)
        ]
        if len(found_terms) > 0:
            result_docs.append(doc_id)
    return result_docs

```