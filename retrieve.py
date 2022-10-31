import prerank
import argparse
import json

from rankers import static, icfwLA

def import_json(path):
    with open(path, 'r') as in_:
        dataSetInfo = json.load(in_)
    return dataSetInfo



def import_json(path):
    with open(path, 'r') as in_:
        dataSetInfo = json.load(in_)
    return dataSetInfo


def rerank(query, index_name, model_name):
    models = {
            'static': static,
            'icfwLA': icfwLA}

    dataSetInfo = import_json('datasetInfo.json')[index_name]
    preranked_data = prerank.retrieve_documents(query, index_name)
    reranked, weights = models[model_name].rerank(preranked_data, dataSetInfo)
    return reranked, weights


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("-index_name", help="name of the index", default='trec-web')
    parser.add_argument("-query", help="query", default='foreign minorities germany')
    parser.add_argument("-model_name", help="model name", default='static')

    args = parser.parse_args()
    index_name = args.index_name
    query = args.query
    model_name = args.model_name
    rerank(query, index_name, model_name)

