# DIS summary 

## Vector space retreival (TF-IDF)
- an example for a tokenizer 

```{python}
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

stemmer = PorterStemmer() 
def tokenize(text):
    """
    It tokenizes and stems an input text.
    
    :param text: str, with the input text
    :return: list, of the tokenized and stemmed tokens.
    """
    text = "".join([ch for ch in text if ch not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    return [stemmer.stem(word.lower()) for word in tokens if word not in stopwords.words('english')]
```
- term frequency for term $i$ in document $j$: 
$$tf(i,j) = \frac{freq(i,j)}{\max_{k \in d_j} freq(k,j)}$$
- inverse document frequency for term $t$. 
$$idf(t) = \log \frac{\# docs}{\# docs\,with\,term\,t}$$
- read the max count from a counter `max_count = counts.most_common(1)[0][1]`
- cosine similarity: 
$$sim(\vec{a}, \vec{b}) = \frac{\vec{a} \cdot \vec{b}}{|\vec{a}|\cdot|\vec{b}|}$$
- tf-idf implementation of sklearn: 
```{python}
from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer(analyzer='word', ngram_range=(1,1), min_df = 1, stop_words = 'english')
features = tf.fit_transform(original_documents)
```
- recall & precision at k: 
  - For recall at k we consider the whole ground truth as basis. For precision we only consider the $k$ elements retreived.
  $$recall_{at \, k} = len(set(predictions[:k]).intersection(set(gt))) / len(gt)$$
  $$precision_{at \, k} = len(set(predictions[:k]).intersection(set(gt))) / k$$

- mean average precision (MAP): 
  - first sum: average over a set of queries
  - second sum: average over the interpolated percision. For that iterate over $k$ in reverse order and add the highest precision value seen so far to the list of precision values IF a correct document is retreived. Normalize by the length of the ground truth set. 
  $$MAP = \frac{1}{|Q|} \sum_{j=1}^{|Q|} \frac{1}{m_j} \sum_{k=1}^{m_j} P_{int}(D_{jk}) $$

## Taxonomy induction 
- Challenge: reduce the noise as much as possible w/o loosing to many good results. In a taxonomy graph with longer paths we have inherenlty more noise. 
- Find all relations in a text with python regex: `re.findall("[a-z]+ is a [a-z]+", file_text)`. 