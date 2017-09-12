import sys
import time
import etri_portnum
import mrf_word_sense_disambiguation


def main():
    global PARSE_PORT_NUM

    file_num = '0' if len(sys.argv) < 2 else str(sys.argv[1])
    etri_portnum.PORT_NUM = 33333 if len(sys.argv) < 2 else int(sys.argv[2])
    print (etri_portnum.PORT_NUM)

    mrf_disambiguater = mrf_word_sense_disambiguation.MRFWordSenseDisambiguation()
    mrf_disambiguater.init()

    print ('data loaded')

    wiki_input_template = "C:\SWRC_DATA\wiki201707\kowiki-20170701-kbox_initial-wikilink-sentences{{num}}"
    wiki_output_template = "C:\SWRC_DATA\wiki201707\kowiki-20170701-kbox_initial-wikilink-sentences-wsd{{num}}"

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

        items = line.strip().split('\t')

        if (len(items) == 4):
            items[-1] = items[-1].replace('<< _sbj_ >>', '<< ' + items[0] + '_s >>')
            items[-1] = items[-1].replace('<< _obj_ >>', '<< ' + items[1] + '_o >>')

        input_sent = items[-1].strip()

        try:
            result, sent = mrf_disambiguater.disambiguate({
                'text' : input_sent,
                'word' : '가',
                'beginIdx' : 0,
                'endIdx' : 1
            }, mode='ALL_WORD')
        except:
            f_write.write(line.strip() + '\n')
            continue

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

            if (entity_open_count < 2 and ('wsd_index' in wsd)):
                word = wsd['text'] + ('다' if (wsd['type'] != 'NNG') else '')

                s_code = result[str(wsd['wsd_index'])]
                candi_list = mrf_disambiguater.corenet_lemma_obj[word]
                candiIdx = 0
                for candi in candi_list:
                    is_Candi = False
                    for korterm in candi['korterm_set']:
                        if (korterm == s_code):
                            is_Candi = True
                            break
                    if is_Candi:
                        s_code = str(candiIdx)
                        break
                    candiIdx += 1

                word = word + '-@-' + str(s_code)
                new_wsd_list[-1]['text'] = word
                new_wsd_list[-1]['is_WSD'] = True

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

        items[-1] = new_sent
        if (len(items) == 4):
            items[-1] = items[-1].replace('<< ' + items[0] + '_s >>', '<< _sbj_ >>')
            items[-1] = items[-1].replace('<< ' + items[1] + '_o >>', '<< _obj_ >>')

        for i in range(len(items)):
            f_write.write(items[i])
            if (i < len(items)-1):
                f_write.write('\t')
        f_write.write('\n')


    f.close()
    f_write.close()

    print ('%d sec time elpased'%(int(time.time())-sttime))

if __name__ == "__main__":
    main()

