import numpy as np

def rankerFunction(fields, query_data, datasetInfo):

    arr_data = query_data['numpy_data']
    doc_ids = arr_data['elastic_ids']
    arr_data = query_data['numpy_data']
    field_scores = arr_data['bm25_scores_arr']
    aggregated_scores = field_scores.sum(axis=1)
    field_weights = field_scores
    score_weights = list(zip(aggregated_scores.tolist(), field_weights.tolist()))
    sorted_weights = [x[1] for x in sorted(score_weights, key=lambda x:x[0], reverse=True)]
    sorted_weights = np.array(sorted_weights)

    score_doc_id = list(zip(doc_ids,aggregated_scores.tolist()))
    sorted_doc_score_lst = sorted(score_doc_id, key=lambda x:x[1])

    return sorted_doc_score_lst, sorted_weights


def rerank(query_data, datasetInfo):

    fields = datasetInfo['fields']
    doc_score_lst, weights = rankerFunction(fields, query_data, datasetInfo)
    doc_score_lst = sorted(doc_score_lst, key=lambda x: x[1], reverse=True)
    return doc_score_lst, weights


