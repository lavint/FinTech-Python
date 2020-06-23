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