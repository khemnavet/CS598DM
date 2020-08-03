import gensim
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import numpy as np

import task7_constant

class Model(object):
    def __init__(self, model_path):
        #self.model = gensim.models.Word2Vec.load(model_path)
        self.model = KeyedVectors.load(model_path)

    def words_closer_than(self, word1, word2, limit):
        # words that are more similar to word1 and not as similar to word2
        # word1 may be a query
        result = {}
        _, positive, negative = self._parse_query(word1)
        if len(positive) == 1 and len(negative) == 0:
            w1 = positive[0]
            result['word'] = ''
        else:
            results = self.model.wv.most_similar(positive, negative, 1)
            w1 = results[0][0]
            result['word'] = w1
        print("words closer to {0} than {1}".format(w1, word2))
        words = self.model.wv.words_closer_than(w1, word2)
        result['words'] = words[0:limit]
        return result

    def auto_complete(self, query, limit, prefix_suffix):
        # query has to be a single word - no spaces
        # prefix_suffix query at start or end of word, 0 = prefix, 1 = suffix
        words = []
        for word in self.model.wv.vocab:
            if (prefix_suffix == task7_constant.PREFIX and word.startswith(query)) or (prefix_suffix == task7_constant.SUFFIX and word.endswith(query)):
                words.append({'word':word, 'count':self.model.wv.vocab[word].count})
        words = sorted(words, key=lambda x: x['count'], reverse=True)
        return words[0:limit]

    def explore(self, query, num_matches):
        print("explore data, query: {0}, num_matched: {1}".format(query, num_matches))
        query_data = QueryData(query)
        query_data.clear_data()
        if len(query):
            query, positive, negative = self._parse_query(query)
            results = self.model.wv.most_similar_cosmul(positive, negative, num_matches)
            for word, dist in results:
                query_data.labels.append(word)
                query_data.distances.append(dist)
                query_data.vectors.append(self.model[word])
            query_data.parsed_positive = positive
            query_data.parsed_negative = negative
            query_data.vocab_size = len(self.model.wv.vocab)
            query_data.query_size = len(query_data.labels)
        return query_data

    def _parse_query(self, query):
        query = gensim.corpora.textcorpus.strip_multiple_whitespaces(query)
        words = query.split(' AND ')
        positive = []
        negative = []
        for word in words:
            if word.startswith('NOT '):
                negative.append(word[4:].lower())
            else:
                positive.append(word.lower())
        return query, positive, negative

class QueryData(dict):
    def __init__(self, query, labels = [], vectors = []):
        self.query = query
        self.labels = labels
        self.distances = []
        self.vectors = vectors
        self.vocab_size = 0
        self.query_size = 0
        self.parsed_positive = []
        self.parsed_negative = []
        self.dim_embedded = []
        self.embedding = []
        self.cluster_data = []
        self.cluster_centroids = []
    
    def clear_data(self):
        self.distances.clear()
        self.labels.clear()
        self.vectors.clear()
        self.parsed_positive.clear()
        self.parsed_negative.clear()
        self.dim_embedded.clear()
        self.cluster_data.clear()
        self.cluster_centroids.clear()
    
    def _closest_node(self, node, nodes):
        nodes = np.asarray(nodes)
        dist_2 = np.sum((nodes - node)**2, axis=1)
        return np.argmin(dist_2)

    def dim_reduce(self):
        self.dim_embedded = TSNE(n_components=2).fit_transform(np.array(self.vectors, dtype=np.float64))
        self.embedding = self.dim_embedded.tolist()

    def cluster(self, num_clusters):
        clustering = KMeans(n_clusters = num_clusters).fit(self.dim_embedded)
        cluster_labels = clustering.labels_.tolist()
        centroids = clustering.cluster_centers_.tolist()
        for cluster_id in range(num_clusters):
            closest_node = self._closest_node(centroids[cluster_id], self.dim_embedded)
            closest_node_word = self.labels[closest_node]
            self.cluster_centroids.append({'x':centroids[cluster_id][0], 'y':centroids[cluster_id][1], 'label':cluster_labels[cluster_id], 'word':closest_node_word})
        embedding = self.dim_embedded.tolist()
        for item in range(len(cluster_labels)):
            self.cluster_data.append({'x': embedding[item][0], 'y': embedding[item][1], 'label': cluster_labels[item]})


    def to_dict(self) :
        result = {'query': self.query, 'labels': self.labels, 'vocab_size': self.vocab_size, 'query_size': self.query_size, 'centroids': self.cluster_centroids, 'cluster_data': self.cluster_data }
        return result
    