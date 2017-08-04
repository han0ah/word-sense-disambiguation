import json
import corenet
import math
import data_util
import operator
import time
from data_manager import DataManager
from sklearn.metrics.pairwise import cosine_similarity


def get_corenet_num(input_vector, word):
    word = '시장'
    matching_def_list = data_util.get_real_corenet_matching_def_list(word)
    debug = 1

def main():
    DataManager.init_data()
    print ("Data loaded")

    input_filepath = "C:\SWRC_DATA\small_data"
    output_filepath = "C:\SWRC_DATA\small_data_output"

    f = open(input_filepath, 'r', encoding='utf-8')
    for line in f:
        etri_obj = json.loads(line)
        sentence = etri_obj['sentence']

        for sent in sentence:
            text = sent['text']
            wsd_list = sent['WSD']
            input_vector = DataManager.tfidf_obj.transform([text])
            new_wsd_list = []
            for wsd in wsd_list:
                corenet_voc_num = -1
                corenet_sem_num = -1
                kortermnum = ""
                if (wsd['type'] == 'NNG' or wsd['type'] == 'VA' or wsd['type'] == 'VV'):
                    word = wsd['text'] + ('다' if (wsd['type'] != 'NNG') else '')
                    get_corenet_num(input_vector, word)

                wsd['corenet_voc_num'] = corenet_voc_num
                wsd['corenet_sem_num'] = corenet_voc_num
                new_wsd_list.append(wsd)

    f.close()

'''
sent['WSD']
 -text
 -type
 -
'''

if __name__ == "__main__":
    main()

