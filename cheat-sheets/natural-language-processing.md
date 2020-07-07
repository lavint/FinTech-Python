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

nltk.download("reuters")

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

wc = WordCloud(width=800, 
               height=600, 
               max_words=28, 
               colormap="RdYlBu").generate(input_words_str)

plt.imshow(wc)
plt.axis("off")
fontdict = {"fontsize": 20, "fontweight": "bold"}
plt.title("Word Cloud", fontdict=fontdict)
plt.show()

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

* The higher the `TF-IDF score`, the more relevant that word is in that particular document

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

<br>
<br>

## <u>***Getting data from News API***</u>

There are 2 main endpoints:
1. Top Headlines: Retrieves breaking news headlines
2. Everything: Retrieves news and articles from over 30,000 different sources
3. Sources: Retrieves info about the most important article sources that this service indexes


*Notes:*

`q` keywords or phrases to search for in the article title and body.

Advanced search is supported here:

* Surround phrases with quotes (") for exact match.
* Prepend words or phrases that must appear with a + symbol. Eg: +bitcoin
* Prepend words that must not appear with a - symbol. Eg: -bitcoin
* Alternatively you can use the AND / OR / NOT keywords, and optionally group these with parenthesis. Eg: crypto AND (ethereum OR litecoin) NOT bitcoin.
* The complete value for q must be URL-encoded.


For more information, visit [News API](https://newsapi.org/docs/client-libraries/python)


```
!pip install newsapi-python
from newsapi import NewsApiClient
api_key = os.getenv("news_api")
newsapi = NewsApiClient(api_key=api_key)



top_headlines = newsapi.get_top_headlines(q='bitcoin',
#                                           sources='bbc-news,the-verge',
                                          category='business',
                                          language='en',
                                          country='us')


all_articles = newsapi.get_everything(q='bitcoin',
#                                       sources='bbc-news,the-verge',
#                                       domains='bbc.co.uk,techcrunch.com',
#                                       from_param='2020-06-04',
#                                       to='2020-07-04',
                                      language='en',
                                      sort_by='relevancy',
#                                       page=2
#                                       page_size=100
                                     )


# Put articles into dataframe
pd.DataFrame.from_dict(all_articles['articles'])


# Create a function to get only certain columns into the dataframe
def create_df(news, language):
    articles = []
    for article in news:
        try:
            title = article["title"]
            description = article["description"]
            text = article["content"]
            date = article["publishedAt"][:10]

            articles.append({
                "title": title,
                "description": description,
                "text": text,
                "date": date,
                "language": language
            })
        except AttributeError:
            pass

    return pd.DataFrame(articles)


btc_df = create_df(all_articles['articles'], 'en')


# Output to csv
file_path = Path("Data/btc_en.csv")
btc_df.to_csv(file_path, index=False, encoding='utf-8-sig')
```


<br>
<br>

## <u>***Sentiment Polarity***</u>
* VADER (Valence Aware Dictionary and Sentiment Reasoner) is a tool used to score the sentiment of human speech as positive, neutral, or negative based on a set of rules and a predefined lexicon (a list of words) that was manually tagged as positive or negative according to semantic orientation

There are 4 scores for each analyzed text:
1. Positive
2. Neutral
3. Negative
4. Compound


* The pos, neu and neg scores ranges from 0 to 1
* The compound score ranges from -1 (most negative) to +1 (most positive): 

    * positive sentiment: compound score >= 0.05
    * neutral sentiment: -0.05 < compound score < 0.05
    * negative sentiment: compound score <= -0.05


```
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

analyzer = SentimentIntensityAnalyzer()

api_key = os.getenv("news_api")
newsapi = NewsApiClient(api_key=api_key)

# Get top 100 articles from newsapi
all_articles = newsapi.get_everything(q='bitcoin',
                                      language='en',
                                      sort_by='relevancy',
                                      page_size=100
                                     )

# Create the bitcoin sentiment scores DataFrame
btc_sentiments = []

for article in all_articles["articles"]:
    try:
        text = article["content"]
        date = article["publishedAt"][:10]
        sentiment = analyzer.polarity_scores(text)
        compound = sentiment["compound"]
        pos = sentiment["pos"]
        neu = sentiment["neu"]
        neg = sentiment["neg"]
        
        btc_sentiments.append({
            "text": text,
            "date": date,
            "compound": compound,
            "positive": pos,
            "negative": neg,
            "neutral": neu
            
        })
        
    except AttributeError:
        pass
    
# Create DataFrame
btc_df = pd.DataFrame(btc_sentiments)

# Reorder DataFrame columns
cols = ["date", "text", "compound", "positive", "negative", "neutral"]
btc_df = btc_df[cols]


# Show stats
btc_df.describe()
```



<br>
<br>

## <u>***Tone Analysis***</u>

1. Non-conversational Tone

    ```
    ! pip install --upgrade "ibm-watson>=3.0.3"
    from pandas import json_normalize
    from ibm_watson import ToneAnalyzerV3
    from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

    # Get the Tone Analyzer API Key and URL
    tone_api = os.getenv("tone_key")
    tone_url = os.getenv("tone_url")

    # Create authentication object
    authenticator = IAMAuthenticator(tone_api)

    # Create tone_analyzer instance
    tone_analyzer = ToneAnalyzerV3(
        version="2020-07-01",
        authenticator=authenticator
    )

    # Set the service endpoint
    tone_analyzer.set_service_url(tone_url)


    # Define text to analyze
    text = """
    Today, I am very happy because it is a holiday. 
    I have been studying for a while so I feel productive.
    However, sitting on the chair causes me much back pain. 
    I need to get up more and stretch!
    """

    # Analyze the text's tone with the 'tone()' method.
    tone_analysis = tone_analyzer.tone(
        {"text": text},
        content_type="application/json",
        content_language="en",
        accept_language="en",
    ).get_result()

    # Display tone analysis results
    print(json.dumps(tone_analysis, indent=2))

    ```

    Document Tones Dataframe
    ```
    doc_tone_df = json_normalize(
        data=tone_analysis["document_tone"], 
        record_path=["tones"])
    ```

    Sentence Tones Dataframe
    ```
    sentences_tone_df = json_normalize(
        data=tone_analysis["sentences_tone"],
        record_path=["tones"],
        meta=["sentence_id", "text"],
    )
    ```

<br>

2. Conversational Tone

    ```
    # Define conversational utterances

    utterances = [
        {"text": "Hello, your product is not working.", "user": "customer"},
        {"text": "OK, Please let me know which part if not working.", "user": "agent"},
        {"text": "Well, nothing is working :(", "user": "customer"},
        {"text": "Sorry to hear that.", "user": "agent"},
    ]

    # Analyze utterances using the 'tone_chat()' method

    utterance_analysis = tone_analyzer.tone_chat(
        utterances=utterances, 
        content_language="en", 
        accept_language="en"
    ).get_result()

    print(json.dumps(utterance_analysis, indent=2))

    ```

    Conversation Tone Dataframe

    ```
    chat_tone_df = json_normalize(
        data=utterance_analysis["utterances_tone"],
        record_path=["tones"],
        meta=["utterance_id", "utterance_text"],
    )
    ```


<br>
<br>

## <u>***spaCY***</u>
* Its core functions depend on language models learned from tagged text instead of programmed rules

* More flexible and more accurate than some of the NLTK tools

* Its language models trade off accuracy for speed, so if the corpus is large, use a simpler rule-based solution (NLTK)


<br>

***Dependency parsing & POS Tagging***

1) Dependency parsing

    * Each sentence is made of not just the words that it contains but also the relationships that are implicit between them

    * Dependency parser is used to make the relationship explicit

2) Part-of-speech tagging

    * Each word in a sentence is designated a grammatical part of speech, such as noun, verb, or adjective

```
import spacy

# For English
nlp = spacy.load("en_core_web_sm")

sentence = 'Someone is sitting in the nice shade today because someone planted seeds a long time ago.'

sen = nlp(sentence)

# Print text, part of speech, dependencies
print([(token.text, token.pos_, token.dep_) for token in sen if token.pos_ == 'NOUN'])

# Print words that describe "shade"
print([token.text for token in sen if (token.head.text == 'shade' and token.pos_ == 'ADJ')])
```

Display relationship in graph
```
from spacy import displacy

# Show the dependency tree
displacy.render(sen, style='dep')
```


<br>

***Named Entity Recognition***
* Extracts specific types of nouns ("named entities") from the text

For more information, visit [spaCY](https://spacy.io/api/annotation#named-entities) site

```
import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")

sen = nlp(u"Mary Kay was a dancer in the National Ballet Association.")

for ent in sen.ents:
    print(ent.text, ent.label_)

displacy.render(sen, style='ent')
```