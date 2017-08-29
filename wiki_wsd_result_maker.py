import pickle
import time
import data_util
import math
from data_manager import DataManager
from sklearn.metrics.pairwise import cosine_similarity

def get_corenet_num(input_vector, word):
    global corenet_lemma_obj

    korterm_list = []
    ttt = corenet_lemma_obj[word]
    for idx in range(len(ttt)):
        item = ttt[idx]
        list = item['korterm_set']
        for korterm in list:
            korterm_list.append({'korterm':korterm, 'idx':idx})

    max_cos_similiarity = -1 * math.inf
    max_korterm = ''
    max_idx = '0'

    for korterm_item in korterm_list:
        korterm = korterm_item['korterm']
        index = korterm_item['idx']
        if (korterm not in DataManager.korenet_tfidf):
            continue
        if (max_korterm == korterm):
            continue

        korterm_vec = DataManager.korenet_tfidf[korterm]
        cos_similarity = cosine_similarity(input_vector, korterm_vec)[0][0]

        if (cos_similarity > max_cos_similiarity):
            max_cos_similiarity, max_korterm, max_idx = cos_similarity, korterm, str(index)

    return max_idx, max_cos_similiarity

def main():
    global corenet_lemma_obj
    corenet_lemma_obj = pickle.load(open('./data/corenet_lemma_info_obj.pickle', 'rb'))
    DataManager.init_data()
    print ('data loaded')

    wiki_input_template = "C:\SWRC_DATA\kowiki-20170701-dump{{num}}.txt"
    wiki_output_template = "C:\SWRC_DATA\kowiki-20170701-dump-out{{num}}.txt"

    sttime = int(time.time())

    input_filepath = wiki_input_template.replace("{{num}}",'0')
    output_filepath = wiki_output_template.replace("{{num}}",'0')

    f = open(input_filepath, 'r', encoding='utf-8')
    f_write = open(output_filepath, 'w', encoding='utf-8')
    index = 0
    for line in f:
        index += 1
        if (index % 100 == 0):
            elapsed = int(time.time()) - sttime
            print("%d  done.  %d sec elapsed" % (index,elapsed))
            sttime = int(time.time())

        #TODO : 시간 표시

        input_sent = line.strip()
        etri_result = data_util.get_nlp_test_result(input_sent)
        if (etri_result is None):
            continue
        sentence = etri_result['sentence']
        for sent in sentence:
            match_korterm_list = []
            # sent_id 찾기
            text = sent['text'].strip()
            wsd_list = sent['WSD']
            input_vector = DataManager.tfidf_obj.transform([text])
            if (len(text) < 20):
                break
            if (len(input_vector.data) == 0):
                is_error = True
                break
            new_wsd_list = []
            for wsd in wsd_list:
                wsd['is_WSD'] = False
                new_wsd_list.append(wsd)
                if (wsd['type'] == 'NNG' or wsd['type'] == 'VA' or wsd['type'] == 'VV'):
                    word = wsd['text'] + ('다' if (wsd['type'] != 'NNG') else '')

                    if (word not in corenet_lemma_obj):
                        continue
                    corenet_list = corenet_lemma_obj[word]
                    if (len(corenet_list) < 1):
                        continue

                    wsd_result, confidence = get_corenet_num(input_vector, word)
                    if (len(wsd_result) < 1):
                        continue
                    if (confidence >= 0.15 or len(corenet_list) == 1):
                        word = word + '-@-' + str(wsd_result)
                        new_wsd_list[-1]['text'] = word
                        new_wsd_list[-1]['is_WSD'] = True
            debug = 1
            new_sent = ""
            word_list = sent['word']
            wsd_idx = 0
            wsd_len = len(new_wsd_list)
            for word_idx in range(len(word_list)):
                word = word_list[word_idx]
                new_word = ''
                iscontain_wsd = False
                while wsd_idx < wsd_len:
                    if (wsd_list[wsd_idx]['begin'] >= word['begin'] and wsd_list[wsd_idx]['end'] <= word['end']):
                        if(wsd_list[wsd_idx]['is_WSD']):
                            iscontain_wsd = True
                            if (len(new_word) > 0):
                                new_word += ' '
                        new_word += wsd_list[wsd_idx]['text']
                    else:
                        break
                    wsd_idx += 1
                if (iscontain_wsd):
                    new_sent += new_word
                else:
                    new_sent += word['text']
                if (word_idx < len(word_list)-1):
                    new_sent += ' '
            f_write.write(new_sent + '\n')

    f.close()
    f_write.close()

    print ('%d sec time elpased'%(int(time.time())-sttime))

if __name__ == "__main__":
    main()

