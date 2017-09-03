import sys
import pickle
import time
import data_util
import math
import etri_portnum
from data_manager import DataManager
from sklearn.metrics.pairwise import cosine_similarity

def main():
    global PARSE_PORT_NUM

    file_num = '0' if len(sys.argv) < 2 else str(sys.argv[1])
    etri_portnum.PORT_NUM = 33333 if len(sys.argv) < 2 else int(sys.argv[2])
    print (etri_portnum.PORT_NUM)

    DataManager.init_data()
    print ('data loaded')


    wiki_input_template = "C:\SWRC_DATA\wiki201707\kowiki-20170701-dump-sentences{{num}}.txt"
    wiki_output_template = "C:\SWRC_DATA\wiki201707\kowiki-20170701-dump-sentences-out{{num}}.txt"

    sttime = int(time.time())

    input_filepath = wiki_input_template.replace("{{num}}",file_num)
    output_filepath = wiki_output_template.replace("{{num}}",file_num)

    f = open(input_filepath, 'r', encoding='utf-8')
    f_write = open(output_filepath, 'w', encoding='utf-8')
    index = 0
    for line in f:
        index += 1
        if (index % 100 == 0):
            elapsed = int(time.time()) - sttime
            print("%d  done.  %d sec elapsed. portnum : %d, docnum : %s" % (index,elapsed,etri_portnum.PORT_NUM,file_num))
            sttime = int(time.time())

        input_sent = line.strip()
        etri_result = data_util.get_nlp_test_result_socket(input_sent)
        if (etri_result is None):
            continue
        sentence = etri_result['sentence']
        for sent in sentence:
            match_korterm_list = []
            # sent_id 찾기
            text = sent['text'].strip()
            wsd_list = sent['WSD']

            new_wsd_list = []
            entity_open_count = 0
            prev_text = ''
            for wsd in wsd_list:
                wsd['is_WSD'] = False
                new_wsd_list.append(wsd)

                if (wsd['text'] == '<'):
                    if (entity_open_count == 0):
                        entity_open_count = 1
                    if (entity_open_count == 1):
                        if (prev_text == '<'):
                            entity_open_count = 2
                        else:
                            entity_open_count = 0
                if (wsd['text'] == '>'):
                    if (entity_open_count == 2):
                        entity_open_count = 1
                    if (entity_open_count == 1):
                        entity_open_count = 0

                prev_text = wsd['text']

                if (entity_open_count  < 2 and (wsd['type'] == 'NNG' or wsd['type'] == 'VA' or wsd['type'] == 'VV')):
                    word = wsd['text'] + ('다' if (wsd['type'] != 'NNG') else '')
                    word = word + '-@-' + str(wsd['scode'])
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

