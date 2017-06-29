import time
import urllib.request
import json
import random
from collections import Counter
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib

token_start_time = 0
tokenize_count = 0

def read_corenet_definition_data():
    '''
      idx : 일련번호
      term : 표제어
      vocnum : 동음 형용사에 대한 표제어 구분 순번
      semnum : 의미번호
      definition1 : 의미풀이
      definition2 : 풀이된말
      usage : 예문
    '''
    f = open('../data/definition.dat', 'r', encoding='utf-8')

    definition_list = []
    i = 0
    sttime = time.time()
    for line in f:
        items = line.strip().split('\t')
        definition_list.append({
            'id': int(items[0]),
            'term': items[1],
            'vocnum': int(items[2]),
            'semnum': int(items[3]),
            'definition1': items[4] if len(items) > 4 else '',
            'definition2': items[5] if len(items) > 5 else '',
            'usuage': items[5] if len(items) > 6 else ''
        })

        i += 1
        if (i % 1000 == 0):
            print(str(i) + 'item finished')

    f.close()
    return definition_list


def get_pos_tag_result(text):
    '''
    주어진 텍스트에 대해서 ETRI 텍스트 분석 결과를 반환한다. 
    '''
    etri_pos_url = 'http://143.248.135.60:31235/etri_pos'
    data = '{"text":"' + text + '"}'
    req = urllib.request.Request(etri_pos_url, data=data.encode('utf-8'))
    result = []
    try:
        response = urllib.request.urlopen(req)
        result = response.read().decode('utf-8')
        result = json.loads(result)
    except:
        print('Error Text : ' + text)
        return result
    return result


def convert_deflist_to_sent_list(definition_list):
    '''
    입력으로 받은 definition 데이터 리스트를 단어들의 집합으로 이루어진 문장 목록으로 변환
    '''
    sent_list = []
    for definiton in definition_list:
        term = definiton['term']
        text = term + '. ' \
               + definiton['definition1'].replace('～',term) + ' ' \
               + definiton['definition2'].replace('～',term) + ' ' \
               + definiton['usuage'].replace('～',term)
        if (len(text) > 0):
            sent_list.append(text)
    return sent_list

def etri_tokenizer(text):
    global tokenize_count, token_start_time
    word_list = []
    pos_tag_result = get_pos_tag_result(text)
    for sent in pos_tag_result:
        morph_list = sent['morp']
        for morph in morph_list:
            if (morph['type'][0] == 'S'): # 부호는 무시
                continue
            word_list.append(morph['lemma'])

    tokenize_count += 1
    if ((tokenize_count % 1000) == 0 ):
        print('%d tokenize finished %.2f second elpased'%(tokenize_count, time.time()-token_start_time))
        token_start_time = time.time()

    return word_list

def construct_tf_idf(defintion_list):
    global token_start_time
    token_start_time = time.time()
    '''
    random.shuffle(defintion_list)
    defintion_list = defintion_list[0:10000]
    sent_list = convert_deflist_to_sent_list(defintion_list)
    whole_word_list = []
    for sent in sent_list:
        word_list = etri_tokenizer(sent)
        whole_word_list.extend(word_list)

    count_whole = Counter(whole_word_list)
    debug = 1
    '''
    sent_list = convert_deflist_to_sent_list(defintion_list)
    tfidf = TfidfVectorizer(tokenizer=etri_tokenizer ,min_df=3, max_df=20000)
    tfidf.fit(sent_list)
    joblib.dump(tfidf, '../data/trained_tfidf.pkl')


def main():
    definition_list = read_corenet_definition_data()
    construct_tf_idf(definition_list)
    print('finished')

if __name__ == "__main__":
    main()