import prerank
import pickle
import argparse
import json
import os

from rerankers import static, icfwLA, icfwG, icfwGA, linear

def import_json(path):
    with open(path, 'r') as in_:
        dataSetInfo = json.load(in_)
    return dataSetInfo

def import_pickle(path):
    with open(path, 'rb') as in_:
        file = pickle.load(in_)
    return file


def dump_pickle(path, file):
    with open(path, 'wb') as out:
        pickle.dump(file, out)


def import_json(path):
    with open(path, 'r') as in_:
        dataSetInfo = json.load(in_)
    return dataSetInfo


def retrieve(query, index_name, model_name):
    models = {
            'static': static,
            'icfwLA': icfwLA,
            'icfwG': icfwG,
            'icfwGA': icfwGA,
            'linear': linear,
            }

    dataSetInfo = import_json('datasetInfo.json')[index_name]
    retrieved_dir = 'retrieved'
    already_retrieved = os.listdir(retrieved_dir)
    save_path = os.path.join(retrieved_dir, '{}-{}-{}.json'.format(query.replace(' ','-' ),index_name,model_name))
    if save_path.split('/')[1] not in already_retrieved:
        preranked_data = prerank.retrieve_documents(query, index_name)
        reranked, weights = models[model_name].rerank(preranked_data, dataSetInfo)
        dump_pickle(save_path, (reranked, weights))
    else:
        reranked, weights = import_pickle(save_path)
    return reranked, weights


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("-index_name", help="name of the index", default='imdb')
    parser.add_argument("-query", help="query", default='Indiana Jones Ford')
    parser.add_argument("-model_name", help="model name", default='icfwLA')

    args = parser.parse_args()
    index_name = args.index_name
    query = args.query
    model_name = args.model_name
    retrieve(query, index_name, model_name)

