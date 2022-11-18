from sklearn.metrics.pairwise import chi2_kernel
import numpy as np

def calc_similarity(fetch_ranking_topk, query, seed_entity, potential_entities, interesting_weights, index_name, model_name, es):
    rank_scores = []
    entity_rel = {}

    for entity in potential_entities:
        new_query = query.replace(seed_entity, entity)
        ranking, weights = fetch_ranking_topk(new_query, index_name, model_name, k=20)
        topSimilarities = []
        for result_weight_vector in weights.tolist():
            result_weight_vector = np.array([result_weight_vector])
            similarityMatrix = chi2_kernel(result_weight_vector, interesting_weights)
            similarityMatrix = np.exp(-1*similarityMatrix)
            topSim = similarityMatrix.sum()
            topSimilarities.append(topSim)
        n = 2
        
        max_N_ind = np.argpartition(topSimilarities, n)[:n]
        if index_name == 'dbpedia':
            top_labels = [es.get(index=index_name, id=ranking[x][0])['_source']['label'] for x in max_N_ind]
        elif index_name == 'imdb':

            top_labels = [es.get(index=index_name, id=ranking[x][0])['_source']['movie_name'] for x in max_N_ind]
        entity_rel[entity] = top_labels
        rank_score = np.array(topSimilarities).max()
        rank_scores.append((entity, rank_score))
    return rank_scores, entity_rel
