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


## Probabilistic modeling
- We want to estimate $P(q|M_d)$ where $M_d$ is the language model generating the document. Using an independence assumption: 
$$P(q|M_d) \approx \prod_{t \in q} P_{mle} (t | M_d)$$
For the term not going to 0 when moddeling $P(q|M_d)$ with a product we introduce smoothing where we also consider the probability of the term being generated in by the collection language model $M_c$. 
$$P(d|q) \approx P(d) \prod_{t\in q} (1-\lambda) P(t|M_c) + \lambda P(t|M_d) $$


## Roccios algo for relevance feedback 
Go in the direction of the average relevant vector and away from the average non relevant vector. A variant sets all nonzero weights of the new query vector to zero. 

$$ \vec{q_m} = \alpha \vec{q_0} + \frac{\beta}{|D_r|} \sum_{\vec{d_j} \in D_r} \vec{d_j} - \frac{\gamma}{|D_{nr}|} \sum_{\vec{d_j} \in D_{nr}} \vec{d_j} $$


## Fagins algorithm for distributed retreival
1. Sort the posting lists by TF-IDF value 
2. Scan all lists (top element of all lists, then second element of all lists ...) until we found $k$ documents in all lists. 
3. Retreive missing weights among the retreived documents. Return the documents with the highest overall weights.


## Inverted list 
An inverted list / file can be created if we map the file to $(doc_id, 'word')$ tuples. Then, reduce by word to a tuple with the word and a list of all occurances $(word, [doc1, doc7, ...])$.


## Latent semantic indexing 
Do an SVD and select the $k$ principle topics for further analysis. 

```{python}
    K,S,Dt = np.linalg.svd(term_doc_matrix, full_matrices=False)

    K_sel = K[:, :k]
    S_sel = np.diag(S[:k])
    Dt_sel = Dt[:k, :]
```


## Word embeddings
Word embeddings can be generated with the `fasttext` library. 
```{python}
model = fasttext.train_unsupervised('epfldocs.txt', model = 'cbow')
vocabulary = model.words
word_embeddings = np.array([model[word] for word in vocabulary])
```


## Taxonomy induction 
- Challenge: reduce the noise as much as possible w/o loosing to many good results. In a taxonomy graph with longer paths we have inherenlty more noise. 
- Find all relations in a text with python regex: `re.findall("[a-z]+ is a [a-z]+", file_text)`. 