

def rankerFunction(fields, query_data):

    arr_data = query_data['numpy_data']
    scores = arr_data['bm25_scores_arr'].sum(axis=1)
    doc_ids = arr_data['doc_ids']
    doc_score_lst = list(zip(doc_ids,scores.tolist()))
    return doc_score_lst, arr_data['bm25_scores_arr']


def rerank(query_data, datasetInfo):

    fields = datasetInfo['fields']
    doc_score_lst, weights = rankerFunction(fields, query_data)
    doc_score_lst = sorted(doc_score_lst, key=lambda x: x[1], reverse=True)
    return doc_score_lst, weights


