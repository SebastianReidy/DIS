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

## HITS 
Compute the update of the authority stores as `auth_new = (np.matmul(np.transpose(A),hub))` and normalize with `auth=auth_new /  np.linalg.norm(auth_new, 2)`. For the hub scores we do the same but w/o transposing $A$. 


## Page rank 
```{python}
def pagerank_iterative(L, R=None):
    if R is None: #We might want to compute R outside this function to avoid recomputing large matrix
        R = np.multiply(L, 1 / np.sum(L,axis=0))
        
    N = R.shape[0]
    e = np.ones(shape=(N,1))
    q = 0.9

    p = e
    delta = 1
    epsilon = 0.001
    i = 0
    while delta > epsilon:
        p_prev = p
        p = np.matmul(q * R, p_prev)
        p = p + (1-q) / N * e
        delta = np.linalg.norm(p-p_prev,1)
        i += 1

    print("Converged after {0} iterations".format(i))
    return R,p
```


## Modularity 
$$Q=\frac{1}{2m} \sum_{i,j} \left ( A_{ij} - \frac{k_i k_j}{2m} \right ) \delta (C_i,C_j)$$
where $C_i, C_j$ are the cluster memberships. $Q>0.3-0.7$ means significant community structure. In the implementation we use $G.number_of_edges(u=None,v=None)$. Edge betweenness is the fraction of shortest paths going over an edge. 


## Transformer models with torch
We have to set the weights to the pretrained embeddings and adjust the internal dimensions if a corresponding error message pops up. `model = AttentionModel(len(train_set.text_vocab), len(train_set.label_vocab), weights=embeddings, e_dim=50)`. We set a batch size and collade the batches such that all samples in the batch have the same dimensions `torch.utils.data.DataLoader(train_set, batch_size=128, collate_fn=model.collate_batch)`. 


## Collaborative filtering 
To get similarity matrices for collaborative filtering use the follwing to get ITEM similarity `item_similarity = 1-pairwise_distances(np.transpose(train_data_matrix), metric='cosine')` w\o the transpose we compute user similarity. Item based filtering is done via (averaging over the weighted items of the same user): 
$$
{r}_{x}(a) =  \frac{\sum\limits_{b \in N_{I}(a)} sim(a, b) r_{x}(b)}{\sum\limits_{b \in N_{I}(a)}|sim(a, b)|}
$$
User based collaborative filtering via (adjusting the average rating of a user, averaging over all users that rated the same item): 
$$
{r}_{x}(a) = \bar{r}_{x} + \frac{\sum\limits_{y \in N_{U}(x)} sim(x, y) (r_{y}(a) - \bar{r}_{y})}{\sum\limits_{y \in N_{U}(x)}|sim(x, y)|}
$$
The `surprise` library offers algorithms for tasks of this kind. 


## Association Rules
Appriori algoirthm: Compute all possible itemsets fulfilling the support requrement. Then compute the confidence of rules (body -> head) of the possible itemsets. Support = $p(body,head)$, confidence = $p(head|body)$. 


## RDF statements 
... have the form subject property object. Where properties define relationships to other resources ore fixed values. RDF allows us to categorize resources into different classes (typing)


## Entity and Information extraction
Get annotations for text with Spacy. 
```{python}
nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
            token.shape_, token.is_alpha, token.is_stop)
    ents = [(e.text, e.label_, e.kb_id_) for e in doc.ents]
```
For example, we can find persons in a text trough `e.label_ == "PERSON"`. We used the heuristic that the first person entity after the word "directed" is the director of a movie. Smoothing of the emission probabilities (no smoothing of transitions): $P_{smooth}(w_i|e_i) = \lambda P(w_i|s_i) + (1-\lambda) \frac{1}{n}$, where $n$ is the number of bigrams / samples. When using a HMM $P(x|start)$ or $P(y|start)$ do not mean the starting states, they denote the prior distributions. `'R'.isupper()` can be used to check if a character is uppercase or not. 
 

## Taxonomy induction 
- Challenge: reduce the noise as much as possible w/o loosing to many good results. In a taxonomy graph with longer paths we have inherenlty more noise. 
- Find all relations in a text with python regex: `re.findall("[a-z]+ is a [a-z]+", file_text)`. 