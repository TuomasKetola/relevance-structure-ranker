import csv
import os
import json


def make_file(file_name, index_name):
    with open(file_name, 'r') as in_:
        reader = csv.reader(in_)
        file = list(reader)
    headers = file[0]
    data = file[1:]
    data  = [x for x in data if x[2]]
    for query in data:
        out_dict = {}
        query_id = query[0]
        out_dict['query_id'] = query[0]
        out_dict['query'] = query[2]
        out_dict['seed_entity'] = query[3]
        out_dict['interesting_documents'] = [x for x in query[6].split(',') if x]
        out_dict['potential_entities'] = [x for x in query[4].split(',') + query[5].split(',') if x]
        pos_qrels = [[query_id, "", x, "2"] for x in query[4].split(',') if x ]
        neg_qrels = [[query_id, "", x, "0"] for x in query[5].split(',') if x ]
        out_dict['qrels'] = pos_qrels + neg_qrels
        with open(os.path.join(index_name, '{}.json'.format(query_id)), 'w') as out:
            json.dump(out_dict,out,indent=2)


if __name__ == '__main__':


    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-file_name", help="csv file path me of the index", required=True)
    parser.add_argument("-index_name", help="index name", required=True)


    args = parser.parse_args()
    file_name = args.file_name
    index_name = args.index_name

    make_file(file_name, index_name)
