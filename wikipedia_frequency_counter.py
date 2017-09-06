import json
import pickle
import time
import math
from data_manager import DataManager
from sklearn.metrics.pairwise import cosine_similarity
import data_util

corenet_lemma_obj = None

def get_corenet_num(input_vector, word):
    global corenet_lemma_obj

    korterm_list = []
    ttt = corenet_lemma_obj[word]
    for item in ttt:
        list = item['korterm_set']
        for korterm in list:
            korterm_list.append(korterm)

    max_cos_similiarity = -1 * math.inf
    max_korterm = ''

    for korterm in korterm_list:
        if (korterm not in DataManager.korenet_tfidf):
            continue
        if (max_korterm == korterm):
            continue

        korterm_vec = DataManager.korenet_tfidf[korterm]
        cos_similarity = cosine_similarity(input_vector, korterm_vec)[0][0]

        if (cos_similarity > max_cos_similiarity):
            max_cos_similiarity, max_korterm = cos_similarity, korterm

    return max_korterm,max_cos_similiarity

def main():
    global corenet_lemma_obj
    DataManager.init_data()
    corenet_lemma_obj = pickle.load(open('./data/corenet_lemma_info_obj_new.pickle','rb'))

    print("Data loaded")

    wiki_input_template = "C:\SWRC_DATA\input_data\Wiki_ETRI_result_{{num}}.txt"

    num_range = [str(i) for i in range(0,5)]

    sent_cnt = 0
    word_cnt = 0
    high_confidence = 0
    sttime = int(time.time())
    real_sttime = int(time.time())
    total_line_count = 0

    korterm_cooccur_freq = {}

    korterm_set = set()

    input_path = './data/corenet/cjkConcept.dat'
    f = open(input_path, 'r', encoding='utf-8')
    idx = 0
    for line in f:
        idx += 1
        if (idx < 15):
            continue
        items = line.strip().split('\t')

        korterm_set.add(items[1])
        korterm_set.add(items[2])
    f.close()
    for korterm1 in korterm_set:
        korterm_cooccur_freq[korterm1] = {}
        for korterm2 in korterm_set:
            korterm_cooccur_freq[korterm1][korterm2] = 0



    for filenum in num_range:
        input_filepath = wiki_input_template.replace("{{num}}",filenum)

        f = open(input_filepath, 'r', encoding='utf-8')
        for line in f:
            total_line_count += 1
            etri_obj = json.loads(line)
            sentence = etri_obj['sentence']
            is_error = False
            for sent in sentence:
                match_korterm_list = []
                # sent_id 찾기
                text = sent['text'].strip()
                wsd_list = sent['WSD']
                input_vector = DataManager.tfidf_obj.transform([text])
                if (len(text) < 10):
                    break
                if (len(input_vector.data) == 0):
                    is_error = True
                    break
                for wsd in wsd_list:
                    if (wsd['type'] == 'NNG' or wsd['type'] == 'VA' or wsd['type'] == 'VV'):
                        word = wsd['text'] + ('다' if (wsd['type'] != 'NNG') else '')

                        if (word not in corenet_lemma_obj):
                            continue
                        corenet_list = corenet_lemma_obj[word]
                        if (len(corenet_list) < 1):
                            continue

                        wsd_result,confidence = get_corenet_num(input_vector, word)
                        if (len(wsd_result) < 1):
                            continue
                        for idx in range(len(corenet_list)):
                            if (wsd_result in corenet_list[idx]['korterm_set']):
                                if (confidence >= 0.14 or len(corenet_list) == 1):
                                    corenet_lemma_obj[word][idx]['frequency'] += 1
                                    high_confidence += 1
                                    match_korterm_list.append(corenet_list[idx]['korterm_set'])
                                word_cnt += 1
                # cooccur frequnecy 측정
                for i in range(0,len(match_korterm_list)):
                    for j in range(0, len(match_korterm_list)):
                        if (i==j):
                            continue
                        for korterm1 in match_korterm_list[i]:
                            for korterm2 in match_korterm_list[j]:
                                korterm_cooccur_freq[korterm1][korterm2] += 1


            if (not is_error):
                sent_cnt += 1
                if (sent_cnt%200 == 0):
                    elapsed = int(time.time()) - sttime
                    print("%d sent done. / %d  word done / %d high confidence /  %d sec elapsed"%(sent_cnt, word_cnt,high_confidence, elapsed))
                    sttime = int(time.time())
            if (total_line_count >= 401198): # 401198
                break
        f.close()
        if (total_line_count >= 401198):
            break

    pickle.dump(corenet_lemma_obj, open('./data/corenet_lemma_info_obj_with_freq_014.pickle', 'wb'))
    pickle.dump(korterm_cooccur_freq, open('./data/korterm_cooccur_freq_014.pickle', 'wb'))
    print("%d sent done. / %d  word done / %d high confidence" % (sent_cnt, word_cnt, high_confidence))

    now_time = int(time.time())
    totaltt = now_time - real_sttime
    print('total %d second elapsed'%totaltt)

if __name__ == "__main__":
    main()

