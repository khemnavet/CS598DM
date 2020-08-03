import flask
from flask import request, jsonify
from flask_cors import CORS

import task7_classes
import task7_constant

app = flask.Flask(__name__)
app.config["DEBUG"] = False
CORS(app)
model = task7_classes.Model('yelp_dataset/word2vecmodel2')

@app.route('/', methods=['GET'])
def home():
    return '<h1>Word2Vec Data Explorer</h1><p>An application to explore a Word2Vec model.</p><ul><li>Auto Complete - autocomplete given string from words in the model vocabulary. /api/auto_complete?q=&ps=0&limit=</li></ul>'

@app.route('/api/close_words', methods=['GET'])
def close_words():
    # /api/close_words?w1=&w2=&limit=
    # word 1 and word 2 are parameters
    # w1 = word 1, required, a string with no spaces or a query string word AND word AND NOT word
    # w2 = word 2, required, a string with no spaces
    # limit = number of words to return, required

    if 'w1' in request.args:
        w1 = request.args['w1']
        w1.strip()
        if len(w1) == 0:
            return jsonify({'error':True, 'result':'The first word is invalid'})
        if w1.find(' ') != -1 and w1.find('AND') == -1: # has spaces but no AND
            return jsonify({'error':True, 'result':'The first word is invalid'})
        if w1.startswith('AND'):
            return jsonify({'error':True, 'result':'The first word is invalid'})
    else:
        return jsonify({'error':True, 'result':'The first word is required'})
    
    if 'w2' in request.args:
        w2 = request.args['w2']
        w2.strip()
        if len(w2) == 0:
            return jsonify({'error':True, 'result':'The second word is invalid'})
        if w2.find(' ') != -1:
            return jsonify({'error':True, 'result':'The second word is invalid'})
    else:
        return jsonify({'error':True, 'result':'The second word is required'})
    
    if 'limit' in request.args:
        try:
            limit = int(request.args['limit'])
            if limit < 0:
                limit = task7_constant.CLOSE_WORDS_LIMIT
        except ValueError:
            limit = task7_constant.CLOSE_WORDS_LIMIT
    else:
        limit = task7_constant.CLOSE_WORDS_LIMIT
    
    try:
        results = model.words_closer_than(w1, w2, limit)
        return jsonify({'error':False, 'result':results})
    except KeyError:
        return jsonify({'error':True, 'result':'No words found'})

@app.route('/api/auto_complete', methods=['GET'])
def auto_complete():
    # /api/auto_complete?q=&ps=0&limit=
    # the query string, prefix_suffix, and limit are parameters
    # q = query string is required, a string with no spaces
    # ps = prefix_suffix is optional integer, default to 0 (0 = prefix, 1 = suffix)
    # limit is optional, will default to 1000

    if 'q' in request.args:
        q = request.args['q']
        q = q.strip()
        if len(q) == 0:
            return jsonify({'error':True, 'result':'The query string is invalid'})
        if q.find(' ') != -1:
            return jsonify({'error':True, 'result':'The query string is invalid'})
    else:
        return jsonify({'error':True, 'result':'The query string is required'})

    if 'ps' in request.args:
        try:
            ps = int(request.args['ps'])
            if ps < task7_constant.PREFIX or ps > task7_constant.SUFFIX:
                ps = task7_constant.PREFIX # default to prefix
        except ValueError:
            ps = task7_constant.PREFIX
    else:
        ps = task7_constant.PREFIX

    if 'limit' in request.args:
        try:
            limit = int(request.args['limit'])
            if limit < 0:
                limit = task7_constant.AUTOCOMPLETE_LIMIT
        except ValueError:
            limit = task7_constant.AUTOCOMPLETE_LIMIT
    else:
        limit = task7_constant.AUTOCOMPLETE_LIMIT
    results = model.auto_complete(q, limit, ps)
    return jsonify({'error':False, 'result':results})

@app.route('/api/explore', methods=['GET'])
def explore():
    # /api/explore?q=&limit=&cluster=&nc=
    # the query string and limit are parameters
    # q = query string, required
    # limit = the number of matching words to return, integer, optional
    # cluster = if to cluster results, integer, optional, can be 0 or 1
    # nc = number of clusters, integer, optional
    if 'q' in request.args:
        q = request.args['q']
        q = q.strip()
        if len(q) == 0:
            return jsonify({'error':True, 'result':'The query string is required'})
        if q.find(' ') != -1 and q.find('AND') == -1: # has spaces but no AND
            return jsonify({'error':True, 'result':'The query string is invalid'})
        if q.startswith('AND'):
            return jsonify({'error':True, 'result':'The query string is invalid'})
    else:
        return jsonify({'error':True, 'result':'The query string is required'})
    
    if 'limit' in request.args:
        try:
            limit = int(request.args['limit'])
            if limit < 0:
                limit = task7_constant.AUTOCOMPLETE_LIMIT
        except ValueError:
            limit = task7_constant.AUTOCOMPLETE_LIMIT
    else:
        limit = task7_constant.AUTOCOMPLETE_LIMIT
    
    if 'cluster' in request.args:
        try:
            cluster = int(request.args['cluster'])
            if cluster < 0 or cluster > 1:
                cluster = task7_constant.CLUSTER_NO
        except ValueError:
            cluster = task7_constant.CLUSTER_NO
    else:
        cluster = task7_constant.CLUSTER_NO

    if 'nc' in request.args:
        try:
            nc = int(request.args['nc'])
            if nc < 0:
                nc = task7_constant.NUMBER_CLUSTERS
        except ValueError:
            nc = task7_constant.NUMBER_CLUSTERS
    else:
        nc = task7_constant.NUMBER_CLUSTERS
    
    try:
        results = model.explore(q, limit)
        if cluster == task7_constant.CLUSTER_YES:
            results.dim_reduce()
            results.cluster(nc)
        return jsonify({'error':False, 'result':results.to_dict()})
    except KeyError:
        return jsonify({'error':True, 'result':'No similar words found'})
    except IndexError:
        return jsonify({'error':True, 'result':'No similar words found'})
    

if __name__ == '__main__':
    # model = task7_classes.Model('yelp_dataset/word2vecmodel2')
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)