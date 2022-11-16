import os
import time
import json
import re

from elasticsearch import Elasticsearch
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

import numpy as np
import pickle


import argparse

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def import_json(path):
    with open(path, 'r') as in_:
        dataSetInfo = json.load(in_)
    return dataSetInfo


def stemText(txt_tokens):
    stemmed_tokens = [ps.stem(t.lower()) for t in txt_tokens]
    stemmed_tokens = [x for x in stemmed_tokens if x not in stop_words]
    return stemmed_tokens


def connectES(password, host):
    es = Elasticsearch(
        host,
        ca_certs="/Users/tuomasketola/Dropbox/phd_files/searchEngineApp/relevance-structure-ranker/certs/http_ca.crt",
        # verify_certs=False,
        basic_auth=("elastic", password)
        )
    return es


def retrieveBM25F(query, fields, es, query_data, index_name):
    """
    function to do bm25f retrieval in order to find out global df etc metrics
    """

    global_avgdl = None
    global_dl = {}
    bm25f_doc_ids = []
    bm25f_scores = []
    global_idfs = {}
    global_dfs = {}
    global_N = None   
    # run BM25f to get the global idf
    bm25f_query_body = {
            'query': {
                  'combined_fields':{
                    'query': query_data['query'],
                    'fields': fields}
                }
            }

    resp = es.search(index=index_name, body = bm25f_query_body, explain=True,size=1000)
    for hit in resp['hits']['hits']:
        explanation = hit['_explanation']
        for term_details in explanation['details']:
            if len(query.split(' ')) == 1:
                term = re.search('\((\w*?)\)', hit['_explanation']['description']).group(1)
                metric_details_lst  = term_details['details']
            else:
                term = re.search('\((\w*?)\)', term_details['description']).group(1)
                metric_details_lst  = term_details['details'][0]['details']

            for metric_details in metric_details_lst:
                if 'idf, computed as log' in metric_details['description']:
                    global_idfs[term] = metric_details['value']
                    for sub_metric_details in metric_details['details']:
                        if 'n, number of docum' in sub_metric_details['description']:
                            global_df = sub_metric_details['value']

                            global_dfs[term] = sub_metric_details['value']
                        elif 'N, total number of doc' in sub_metric_details['description']:
                            global_N = sub_metric_details['value']
                elif 'tf, computed as freq' in metric_details['description']:

                    for sub_metric_details in metric_details['details']:
                        if 'avgdl, average' in sub_metric_details['description']:
                            global_avgdl = sub_metric_details['value']
                        elif 'dl, length of fiel' in sub_metric_details['description']:
                            dlFull = sub_metric_details['value']

    query_data.update({'global_idfs': global_idfs})
    query_data.update({'global_dfs': global_dfs})
    query_data.update({'global_N': global_N})
    query_data.update({'global_avgdl': global_avgdl})
    query_data.update({'global_dl': global_dl})
    bm25f_ids = [x['_id'] for x in resp['hits']['hits']]
    query_data['bm25f_doc_ids'] = bm25f_ids


def retrieveBM25FSA(query, fields, es, query_data, index_name):
    result_set = {}

    global_avgfl = {field:0 for field in fields}
    uniqueIdDict = {
            'homedepot':'product_uid',
            'trec-web': 'docno',
            'dbpedia': 'id',
            'imdb': 'movie_id'}
        
    for feature in fields:
        query_body = {
                'query': {
                      'multi_match':{
                        'query': query_data['query'],
                        'fields': [feature]}
                    }
                }

        bm25f_ids = query_data['bm25f_doc_ids']

        query_body_2 = {
                  "query": {
                        "bool": {
                            "must": [
                                {
                                    "multi_match": {
                                        "query": query_data['query'],
                                        "fields": [feature]
                                                    }
                                                },
                                    ],
                             "filter": {
                                    "ids": {
                                    "values": bm25f_ids}
                                                        
                                        }
                                }
                            }
                  }
        
        resp = es.search(index=index_name, body = query_body, explain=True,size=1000)
        resp2 = es.search(index=index_name, body = query_body_2, explain=True,size=1000, request_timeout=30)
        # query data
        query_str = query
        # normal docs 
        q_doc_lst = []
        query_id  = 'nan'
        for hit in resp['hits']['hits']:
            doc_id = hit['_source'][uniqueIdDict[index_name]]
            q_doc_lst.append((query_id, doc_id))
            score = hit['_score']
            try:
                result_set[doc_id]
            except KeyError:
                result_set[doc_id] = {
                        }
            elastic_id = hit['_id']
            result_set[doc_id][feature] = {
                                            'score':0.0,
                                            'elastic_id': elastic_id,
                                            'terms':[],
                                            'term_tfs':{},
                                            'term_idfs': {},
                                            'avgfl': global_avgfl[feature],
                                            'fl': global_avgfl[feature]
                                            }               

            result_set[doc_id][feature]['score'] = score

            explanation = hit['_explanation']
            for term_details in explanation['details']:
                # import pdb;pdb.set_trace()

                try:
                    term = re.search('weight\({}:(.*?) '.format(feature), term_details['description']).group(1)
                    metric_details_lst  = term_details['details'][0]['details']
                except AttributeError:
                    term = re.search('weight\({}:(.*?) '.format(feature), explanation['description']).group(1)
                    metric_details_lst  = term_details['details']
                result_set[doc_id][feature]['terms'].append(term)
                for metric_details in metric_details_lst:

                    if 'idf, computed as' in metric_details['description']:
                        term_idf = metric_details['value']
                        query_data['q_term_idfs'][feature].update({term:term_idf})
                        result_set[doc_id][feature]['term_idfs'].update({term:term_idf})
                        for sub_metric_details in metric_details['details']:
                            if 'n, number of documents co' in sub_metric_details['description']:
                                term_df = sub_metric_details['value']
                                query_data['q_term_dfs'][feature].update({term:term_df})
                            elif 'N, total number of' in sub_metric_details['description']:
                                Nf = sub_metric_details['value']
                                query_data['fieldNs'][feature] = Nf

                    elif 'tf, computed as freq' in metric_details['description']:
                        for sub_metric_details in metric_details['details']:
                            if 'freq, occurrences' in sub_metric_details['description']:
                                field_tf = sub_metric_details['value']
                                result_set[doc_id][feature]['term_tfs'].update({term:field_tf})
                            elif 'dl, length of field' in sub_metric_details['description']:
                                fl = sub_metric_details['value']

                                result_set[doc_id][feature]['fl'] = fl
                            elif 'avgdl, average' in sub_metric_details['description']:
                                avgfl = sub_metric_details['value']
                                result_set[doc_id][feature]['avgfl'] = avgfl
                                global_avgfl[feature] = avgfl

        # bm25f ids            
        for hit in resp2['hits']['hits']:
            doc_id = hit['_source'][uniqueIdDict[index_name]]
            score = hit['_score']
            if (query_id, doc_id) in q_doc_lst:
                continue

            try:
                result_set[doc_id]
            except KeyError:
                result_set[doc_id] = {
                        }
            elastic_id = hit['_id']
            result_set[doc_id][feature] = {
                                            'elastic_id':elastic_id,
                                            'score':0.0,
                                            'terms':[],
                                            'term_tfs':{},
                                            'term_idfs': {},
                                            'avgfl': global_avgfl[feature],
                                            'fl': global_avgfl[feature]
                                            }               

            result_set[doc_id][feature]['score'] = score

            if score == 0.0:
                import pdb;pdb.set_trace()

            explanation = hit['_explanation']
            for term_details in explanation['details'][0]['details']:

                try:
                    term = re.search('weight\({}:(.*?) '.format(feature), term_details['description']).group(1)
                    metric_details_lst  = term_details['details'][0]['details']
                except AttributeError:
                    try:
                        term = re.search('weight\({}:(.*?) '.format(feature), explanation['description']).group(1)
                        metric_details_lst  = term_details['details']
                        print('here')
                    except AttributeError:
                        term = re.search('weight\({}:(.*?) '.format(feature), explanation['details'][0]['description']).group(1)
                        metric_details_lst  = term_details['details']
                result_set[doc_id][feature]['terms'].append(term)
                for metric_details in metric_details_lst:

                    if 'idf, computed as' in metric_details['description']:
                        term_idf = metric_details['value']
                        query_data['q_term_idfs'][feature].update({term:term_idf})
                        result_set[doc_id][feature]['term_idfs'].update({term:term_idf})
                        for sub_metric_details in metric_details['details']:
                            if 'n, number of documents co' in sub_metric_details['description']:
                                term_df = sub_metric_details['value']
                                query_data['q_term_dfs'][feature].update({term:term_df})
                            elif 'N, total number of' in sub_metric_details['description']:
                                Nf = sub_metric_details['value']
                                if not Nf:
                                    import pdb;pdb.set_trace()
                                query_data['fieldNs'][feature] = Nf

                    elif 'tf, computed as freq' in metric_details['description']:
                        for sub_metric_details in metric_details['details']:
                            if 'freq, occurrences' in sub_metric_details['description']:
                                field_tf = sub_metric_details['value']
                                result_set[doc_id][feature]['term_tfs'].update({term:field_tf})
                            elif 'dl, length of field' in sub_metric_details['description']:
                                fl = sub_metric_details['value']

                                result_set[doc_id][feature]['fl'] = fl
                            elif 'avgdl, average' in sub_metric_details['description']:
                                avgfl = sub_metric_details['value']
                                result_set[doc_id][feature]['avgfl'] = avgfl

    query_terms = query.split(' ')
    for doc_id, results in result_set.items():
        for feat in fields:
            if feat not in results.keys():
                result_set[doc_id][feat] = {
                                            'score':0,
                                            'terms':[],
                                            'term_tfs':{},
                                            'term_idfs': {},
                                            'avgfl': global_avgfl[feat],
                                            'fl': global_avgfl[feat]
                                            }               
                
        for feat in fields:
            for term in query_terms:
                if term not in result_set[doc_id][feat]['term_tfs'].keys():
                    result_set[doc_id][feat]['term_tfs'][term] = 0.0

    query_data.update({'results': result_set})
 

def make_numpy_arrs(query_data, index_name):
    # make array based data dump
    field_dict = {
                'trec-web':['title', 'body'],
                 'homedepot':['product_name', 'product_description','product_attributes'],
                 'dbpedia': ["label" , "attributes" , "categories" , "related_entities", "similar_entities"],
                 'imdb': ["plot", "movie_name", "movie_languages", "movie_countries", "movie_genres", "actors", "characters", "actor_genders"]
                         }
    avg_fl_dict = {
                field: None for field in field_dict[index_name]
                }
    fields = field_dict[index_name]
    datasetInfo = {
                'dbpedia':{
                    "empty_fields": {
                        "similar_entities": 1869980,
                        "label": 940,
                        "attributes": 0,
                        "categories": 41688,
                        "related_entities": 1832508,
                        "all": 0
                        },
                    "total_doc_count":4641889
                    },
                'trec-web':{
                    "empty_fields": {
                        "title": 3062,
                        "body": 361,
                        "all": 0
                    },
                    "total_doc_count":212591
                },
                'homedepot': {
                    "empty_fields": {
                        "product_uid": 0,
                        "product_name": 0,
                        "product_description": 0,
                        "product_attributes": 16263,
                        "all": 0
                    },
                    "total_doc_count": 54668,
                },
                'imdb': {
                    "total_doc_count":42303,
        "empty_fields": {
                     "plot": 0,
                     "movie_name": 99,
                     "movie_languages": 99,
                     "movie_countries": 99,
                     "movie_genres": 99,
                     "actors": 4550,
                     "characters": 19235,
                     "actor_genders": 4816}
            }
        }


    field_idfs = query_data['q_term_idfs']
    field_dfs = query_data['q_term_dfs']
    field_Ns = {
            field: datasetInfo[index_name]['total_doc_count'] - datasetInfo[index_name]['empty_fields'][field]
            for field in fields}

    query_data['numpy_data'] = {}
    terms = [x.lower() for x in query_data['query'].split(' ')]
    field_tfs = {field: [] for field in fields}
    field_lengths = {field: [] for field in fields}
    field_scores = {field: [] for field in fields}

    results = query_data['results']
    doc_ids = []
    elastic_ids = []
    bm25f_score_lst = []
    concat_dls = []
    for document, data in results.items():
        doc_ids.append(document)
        for field in fields:
            if 'elastic_id' in data[field].keys():
                elastic_ids.append(data[field]['elastic_id'])
                break
            else:
                pass

        for field in fields:
            doc_field_data = data[field]
            tf_vect = [doc_field_data['term_tfs'][term]  for term in terms]
            fl = doc_field_data['fl']
            score = doc_field_data['score']
            field_tfs[field].append(tf_vect)
            field_lengths[field].append(fl)
            field_scores[field].append(score)
        
            if doc_field_data['avgfl']:
                avg_fl_dict[field] = doc_field_data['avgfl']
        # if document == '260-1379':
            # import pdb;pdb.set_trace()


            

    field_tfs = np.dstack(list(field_tfs.values()))
    field_scores_= np.array(list(field_scores.values())).T

    # doc lenghths
    field_lengths = np.array(list(field_lengths.values())).T
    field_lengths = field_lengths.reshape((len(results),1,len(fields)))

    # idfs
    idfs = np.dstack(
    [
                [field_idfs[field][t] if t in field_idfs[field].keys() else 0.0 for t in terms]
                        for field in fields
                            ])

    # global idfs
    global_idfs = np.array([
            query_data['global_idfs'][term] if term in query_data['global_idfs'].keys() else 0 for term in terms
            ]).reshape(1,len(terms))

    # global dfs 
    global_dfs = np.array([
            query_data['global_dfs'][term] if term in query_data['global_dfs'].keys() else 0 for term in terms
            ]).reshape(1,len(terms))

    # dfs
    dfs =  np.dstack(
    [
                [field_dfs[field][t] if t in field_dfs[field].keys() else 0.0 for t in terms]
                        for field in fields
                            ])

    # term occcs
    term_occurrences = (field_tfs > 0).astype(int)

    # field occurrences
    field_occurrences = term_occurrences.sum(axis=2).reshape(len(results), len(terms), 1)

    # field Ns
    Nfs = np.array(list(field_Ns.values())).reshape(1,1,len(fields))

    # P of t given F
    #     print(field_Ns)
    p_t_f = term_occurrences * (dfs / Nfs)
    p_t_f[p_t_f == 0] = 1.0


    # P of t given d
    p_t_d = field_occurrences / len(fields)
    p_t_d[p_t_d == 0] = 1

    query_data['numpy_data']['tf_arr'] = field_tfs
    query_data['numpy_data']['fl_arr'] = field_lengths
    query_data['numpy_data']['bm25_scores_arr'] = field_scores_
    query_data['numpy_data']['idf_arr'] = idfs
    query_data['numpy_data']['df_arr'] = dfs
    query_data['numpy_data']['global_idf_arr'] = global_idfs
    query_data['numpy_data']['global_df_arr'] = global_dfs
    query_data['numpy_data']['df_arr'] = dfs
    query_data['numpy_data']['t_occs'] = term_occurrences
    query_data['numpy_data']['f_occs'] = field_occurrences
    query_data['numpy_data']['Nfs'] = Nfs
    query_data['numpy_data']['p_t_f'] = p_t_f
    query_data['numpy_data']['p_t_d'] = p_t_d
    query_data['numpy_data']['doc_ids'] = doc_ids
    query_data['numpy_data']['elastic_ids'] = elastic_ids
    query_data['numpy_data']['avgfl'] = np.array(list(avg_fl_dict.values())).reshape(1,1,len(fields))


def retrieve_documents(query, index_name):
    query = query.lower()

    # initialize query_data
    dataSetInfo = import_json('datasetInfo.json')
    fields = dataSetInfo[index_name]['fields']
    # query_terms = stemText(query.split(' '))
    query_terms = [x.lower() for x in query.split(' ')]
    query_data = {
            'query':' '.join(query_terms),
            'q_term_idfs': {field:{} for  field in fields},
            'fieldNs': {field:None for  field in fields},
            'q_term_dfs': {field:{} for  field in fields} 
            }

    # connect to es
    ES_PASSWORD = '8kDCKZ2ZwFhRAQmFy6JP'
    es = connectES(ES_PASSWORD, "https://localhost:9200")

    if es.ping():
        # print('ES instance running succesfully')
        pass
    else:
        print('ping did not work')

    retrieveBM25F(query, fields, es, query_data, index_name)

    retrieveBM25FSA(query,fields, es, query_data, index_name)
    make_numpy_arrs(query_data, index_name)
    return query_data


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("-index_name", help="name of the index", default='trec-web')
    parser.add_argument("-query", help="query", default='foreign minorities germany')

    args = parser.parse_args()
    index_name = args.index_name
    query = args.query
    retrieve_documents(query, index_name)


