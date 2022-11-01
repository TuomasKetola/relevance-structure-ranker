import numpy as np

def rankerFunction(fields, query_data, datasetInfo):

    # calculate lambda
    arr_data = query_data['numpy_data']
    field_scores = arr_data['bm25_scores_arr']
    doc_ids = arr_data['elastic_ids']
    df_arr = arr_data['df_arr']
    idf_arr = arr_data['idf_arr']
    empty_fields = datasetInfo['empty_fields']
    N_total = datasetInfo['total_doc_count']
    N = [N_total - empty_fields[field] for field in fields]
    N = np.array(N)
    nr_fields = len(fields)

    nr_terms = df_arr.shape[1]
    if nr_terms == 1:
        shape = (nr_fields.shape[0],1 )
        lambda_= np.zeros(shape)
        return lambda_, None
    max_dfs = df_arr.mean(axis=1).max()
    min_dfs = df_arr[df_arr!=max_dfs].mean()
    idf_ratio = idf_arr.mean(axis=1).max()  / idf_arr[idf_arr!=idf_arr.min()].mean()
    Z = idf_ratio

    numerator = np.log((max_dfs*N**Z) / ((min_dfs**Z)*N))
    denominator = np.log((2**(2*Z)*nr_fields**(Z+1)) / (nr_fields **(2*Z)))
    lambda_ = numerator / denominator

    lambda_[np.isnan(lambda_)] = 0
    lambda_[np.isinf(lambda_)] = 0

    # calculate weights
    p_t_d = arr_data['p_t_d']
    p_t_f = arr_data['p_t_f']
    inf_f_d = -np.log(p_t_d.prod(axis=1))
    inf_f_F = -np.log(p_t_f.prod(axis=1))

    field_weights = inf_f_F + inf_f_d * lambda_
    weighted_arr = field_weights * field_scores
    
    aggregated_scores = weighted_arr.sum(axis=1)
    
    doc_ids = arr_data['elastic_ids']
    doc_score_lst = list(zip(doc_ids,aggregated_scores.tolist()))

    doc_score_lst = sorted(doc_score_lst, key=lambda x: x[1], reverse=True)
    return doc_score_lst, field_weights


def rerank(query_data, datasetInfo):

    fields = datasetInfo['fields']
    doc_score_lst, weights = rankerFunction(fields, query_data, datasetInfo)
    doc_score_lst = sorted(doc_score_lst, key=lambda x: x[1], reverse=True)
    return doc_score_lst, weights


