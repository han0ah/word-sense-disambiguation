import json
import pickle
import time
from data_manager import DataManager
from sklearn.metrics.pairwise import cosine_similarity

corenet_obj = None

def get_corenet_num(input_vector, word):
    global corenet_obj

    if (word not in corenet_obj):
        return None
    matching_def_list = corenet_obj[word]
    if (len(matching_def_list) < 1):
        return None

    result_corenet = None
    max_cos_similarity = 0.0
    for cornet_def in matching_def_list:
        if (len(cornet_def['definition1']) < 1 and len(cornet_def['usuage']) < 1):
            continue
        cos_similarity = cosine_similarity(input_vector, cornet_def['vector'])[0][0]
        if (cos_similarity > max_cos_similarity):
            result_corenet, max_cos_similarity = cornet_def, cos_similarity

    if (result_corenet is not None):
        result_corenet['confidence'] = round(max_cos_similarity,3)
    return result_corenet

def main():
    global corenet_obj
    DataManager.init_data()
    corenet_obj = pickle.load(open('./data/corenet_obj.pickle','rb'))


    etri_sent_list_path = "C:\SWRC_DATA\etri_input.txt"
    etri_sentences = []
    f = open(etri_sent_list_path,'r',encoding='utf-8')
    for line in f:
        etri_sentences.append(line.strip())
    f.close()
    print("Data loaded")

    wiki_input_template = "C:\SWRC_DATA\input_data_small\Wiki_ETRI_result_{{num}}.txt"
    wiki_output_template = "C:\SWRC_DATA\output_data_small\Wiki_ETRI_result_{{num}}.txt"

    num_range = [str(i) for i in range(0,2)]

    etri_matching_idx = 0
    cnt_etri_sentence = len(etri_sentences)
    done_cnt = 0
    sttime = int(time.time())

    for filenum in num_range:
        input_filepath = wiki_input_template.replace("{{num}}",filenum)
        output_filepath = wiki_output_template.replace("{{num}}",filenum)


        f = open(input_filepath, 'r', encoding='utf-8')
        f_write = open(output_filepath, 'w', encoding='utf-8')
        for line in f:
            etri_obj = json.loads(line)
            sentence = etri_obj['sentence']
            idx = 0
            for sent in sentence:

                # sent_id 찾기
                text = sent['text']
                while etri_matching_idx < cnt_etri_sentence:
                    if (text == etri_sentences[etri_matching_idx]):
                        break
                    etri_matching_idx += 1
                etri_obj['sentence'][idx]['sent_id'] = (etri_matching_idx + 1)

                wsd_list = sent['WSD']
                input_vector = DataManager.tfidf_obj.transform([text])
                new_wsd_list = []
                for wsd in wsd_list:
                    corenet_voc_num = -1
                    corenet_sem_num = -1
                    confidence = 0.0
                    kortermnum = ""
                    if (wsd['type'] == 'NNG' or wsd['type'] == 'VA' or wsd['type'] == 'VV'):
                        word = wsd['text'] + ('다' if (wsd['type'] != 'NNG') else '')
                        wsd_result = get_corenet_num(input_vector, word)
                        if (wsd_result is not None):
                            corenet_voc_num = wsd_result['vocnum']
                            corenet_sem_num = wsd_result['semnum']
                            kortermnum = wsd_result['kortermnum']
                            confidence = wsd_result['confidence']

                    wsd['corenet_voc_num'] = corenet_voc_num
                    wsd['corenet_sem_num'] = corenet_sem_num
                    wsd['corenet_confidence'] = confidence
                    wsd['kortermnum'] = kortermnum
                    new_wsd_list.append(wsd)

                etri_obj['sentence'][idx]['WSD'] = new_wsd_list
                idx += 1

            f_write.write(json.dumps(etri_obj, ensure_ascii=False) + '\n')
            done_cnt += 1
            if (done_cnt%10 == 0):
                elapsed = int(time.time()) - sttime
                print("%d done. %d sec elapsed"%(done_cnt, elapsed))
                sttime = int(time.time())

        f.close()
        f_write.close()

    print ('%d sec time elpased'%(int(time.time())-sttime))

if __name__ == "__main__":
    main()

