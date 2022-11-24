from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def make_new_query(old_query, seed_entity, new_entity):
    pass

def make_new_query(old_query, seed_entity, new_entity):
    if len(seed_entity.split(' ')) > 1:
        if len(new_entity.split(' ')) > 1:
            new_query = old_query.replace(seed_entity.replace(' ',' AND '), new_entity.replace(' ',' AND '))
        else:
            new_query = old_query.replace('('+seed_entity.replace(' ',' AND ')+')', new_entity)
    else:
        if len(new_entity.split(' ')) > 1: 
            new_query = old_query.replace(seed_entity, '(' + new_entity.replace(' ',' AND ')+')')
        else:
            new_query = old_query.replace(seed_entity, new_entity)
    return new_query


def calc_similarity(fetch_ranking_topk, query, seed_entity, potential_entities, interesting_weights, index_name, model_name, es):
    rank_scores = []
    entity_rel = {}

    for entity in potential_entities:
        new_query = make_new_query(query,seed_entity, entity)
        # new_query = query.replace(seed_entity, entity)
        ranking, weights = fetch_ranking_topk(new_query, index_name, model_name, k=40)
        topSimilarities = []
        for result_weight_vector in weights.tolist():
            result_weight_vector = np.array([result_weight_vector])
            similarityMatrix = cosine_similarity(result_weight_vector, interesting_weights)
            topSim = similarityMatrix.max()
            topSimilarities.append(topSim)
        n = 4
        max_N_ind = np.argpartition(topSimilarities, -n)[-n:]
        if index_name == 'dbpedia':
            top_labels = [es.get(index=index_name, id=ranking[x][0])['_source']['label'] for x in max_N_ind]
        elif index_name == 'imdb':

            top_labels = [es.get(index=index_name, id=ranking[x][0])['_source']['movie_name'] for x in max_N_ind]
        entity_rel[entity] = top_labels
        rank_score = np.array([topSimilarities[x] for x in max_N_ind]).sum()

        rank_scores.append((entity, rank_score))
    
    return rank_scores, entity_rel
