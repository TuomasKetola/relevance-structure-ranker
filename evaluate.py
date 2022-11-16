import prerank
from elasticsearch import Elasticsearch
import os
import argparse
import json
import csv
import numpy as np
import retrieve
from similarity import cosineSim, absoluteSim
from sklearn.metrics import ndcg_score
from sklearn.metrics import average_precision_score
from collections import defaultdict

def fetch_ranking_topk(query, index_name, model_name, k=100):
    ranking, weights = retrieve.retrieve(query, index_name, model_name)
    return ranking[:k], weights[:k]


def connectES(password, host):
    es = Elasticsearch(
    host,
    ca_certs="/Users/tuomasketola/Dropbox/phd_files/searchEngineApp/relevance-structure-ranker/certs/http_ca.crt",
    basic_auth=("elastic", password)
        )
    return es


def calc_ndcg(query_id, ranking, q_qrels,k):
    true_relevance = []
    prediction = []
    qrel_doc_ids = [x[2] for x in q_qrels]
    prediction_doc_ids = list(ranking.keys())
    for qrel in q_qrels:
        doc_id = qrel[2]
        true_relevance.append(float(qrel[3]))
        try:
            prediction.append(ranking[doc_id])
        except KeyError:
            prediction.append(0)

    for doc_id in list(set(prediction_doc_ids) - set(qrel_doc_ids)):
        prediction.append(ranking[doc_id])
        true_relevance.append(0)
    try:
        acc = ndcg_score(np.array([true_relevance]), np.array([prediction]),k=100)
        acc = round(acc,5)
    except:
        import pdb;pdb.set_trace()
        print('something wrong with ndcg calculation')
        exit()
    return acc


def calc_ap(query_id, q_ranking, q_qrels):
    true_relevance = []
    prediction = []
    qrel_doc_ids = [x[2] for x in q_qrels]
    prediction_doc_ids = list(q_ranking.keys())
    no_info_count = 0
    fail = False
    for qrel in q_qrels:
        doc_id = qrel[2]
        true_relevance.append(float(qrel[3]))
        try:
            prediction.append(q_ranking[doc_id])
        except KeyError:
            prediction.append(0)
    len_rel = len(prediction)
    for doc_id in list(set(prediction_doc_ids) - set(qrel_doc_ids)):
        prediction.append(q_ranking[doc_id])
        true_relevance.append(0)
    true_relevance = [0 if x < 1 else 1 for x in true_relevance]
    trues = np.array(true_relevance)
    preds = np.array(prediction)
    ap = average_precision_score(trues, preds)
    return ap


def dump_json(path, file):
    with open(path, 'w') as out:
        json.dump(file, out, indent=2)


def dump_csv(path, file, delimiter=','):
    with open(path, 'w') as out:
        writer = csv.writer(out,delimiter=delimiter)
        writer.writerows(file)


def create_if_not_exsists(dir_):
    if not os.path.exists(dir_):
        os.makedirs(dir_)


def evaluate(index_name, model_name, simModelName, query_file):
    similarityModeldict = {
            "cosineSim": cosineSim,
            "absoluteSim": absoluteSim
            }

    ES_PASSWORD = '8kDCKZ2ZwFhRAQmFy6JP'
    es = connectES(ES_PASSWORD, "https://localhost:9200")
    if es.ping():
    #     print('ES instance running succesfully')
        pass
    else:
        print('ping did not work')

    # retrieve(query, index_name, model_name):
    query_dir = os.path.join('eval_queries', index_name)

    accuracies = {'ndcg': [], 'ap': []}
    if query_file:
        query_files  = [query_file.split('/')[-1]]
    else:
        query_files = os.listdir(query_dir)
    for file_name in query_files:
        file = retrieve.import_json(os.path.join(query_dir, file_name))
        query_id = file['query_id']
        query = file['query']
        seed_entity = file['seed_entity']
        seed_entity_query = query + ' ' +seed_entity
        qrels = file['qrels']
        potential_entities = file['potential_entities']
        interesting_documents = file['interesting_documents']
        ranking, weights = fetch_ranking_topk(seed_entity_query, index_name, model_name, k=100)
        ranking_ids = [x[0] for x in ranking]
        try:
            interesting_weights = [weights[ranking_ids.index(x)] for x in interesting_documents]
        except:
            import pdb;pdb.set_trace()

        rank_scores, entity_rel = similarityModeldict[simModelName].calc_similarity(
                fetch_ranking_topk,
                seed_entity_query, 
                seed_entity, 
                potential_entities, 
                interesting_weights, 
                index_name, 
                model_name,
                es)
        rank_scores = sorted(rank_scores, key= lambda tup:tup[1], reverse=True)
        rank_scores = [(x[0], round(x[1],4)) for x in rank_scores]
        rank_scores_dict = dict(rank_scores)
        ndcg = calc_ndcg(query_id, rank_scores_dict, qrels, 100)
        ap = calc_ap(query_id, rank_scores_dict, qrels)
        accuracies['ndcg'].append(ndcg)
        accuracies['ap'].append(ap)

        query_based_dir = os.path.join('query_based_acc', index_name)
        create_if_not_exsists(query_based_dir)
        save_path = os.path.join(query_based_dir, query_id+'.json')

        save_path = os.path.join(query_based_dir, '{}-{}-{}.json'.format(query_id, model_name,simModelName))
        dump_json(save_path, {'ndcg': ndcg, 'ap': ap})
        print(query, 'ndcg: ', ndcg, '---', "ap: ", ap)
        ranking_dir = os.path.join('rankings',index_name)
        create_if_not_exsists(ranking_dir)

        save_path = os.path.join(ranking_dir, '{}-{}-{}.json'.format(query_id, model_name,simModelName))
        dump_csv(save_path,rank_scores, delimiter='\t')
    accuracy_dir = os.path.join('accuracies', index_name)
    create_if_not_exsists(accuracy_dir)
    save_path = os.path.join(accuracy_dir, '{}-{}.json'.format(model_name,simModelName))
    dump_json(save_path, {'ndcg': round(np.mean(accuracies['ndcg']),4), 'map':round(np.mean(accuracies['ap']),4)})


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("-index_name", help="name of the index", default='imdb')
    parser.add_argument("-model_name", help="model name", default='icfwLA')
    parser.add_argument("-simModelName", help="similarity model name", default='cosineSim')
    parser.add_argument("-query_file", help="path to query file", default='')

    args = parser.parse_args()
    index_name = args.index_name
    model_name = args.model_name
    simModelName = args.simModelName
    query_file = args.query_file
    evaluate(index_name, model_name, simModelName, query_file)

