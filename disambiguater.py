import corenet
import mrf_word_sense_disambiguation
import math
import data_util
import random
import operator
import time
from data_manager import DataManager
from sklearn.metrics.pairwise import cosine_similarity

class Disambiguater:
    '''
    문장과 문장 내의 특정 단어가 주어졌을 때, 
    이 단어의 문맥 상에서의 의미와 일치하는 CoreNet 상의 표제어 및 어깨번호(의미번호) 를 반환하는 기능을 수행하는 모듈들의
    추상 클래스. 각 Disambiguater 모듈들은 이 모듈을 상속받아서 정의된 interface를 구현한다.
    '''
    def disambiguate(self, input):
        '''
        주어젠 입력에 알맞은 표제어 및 어깨번호를 반환한다.
        Input : Dictionary
        -- text : 주어진 문장
        -- word : 주어진 문장에서 CoreNet 상에 알맞은 정보를 찾고자 하는 어휘
        -- beginIdx : 해당 word가 주어진 문장에서 시작하는 위치
        -- endIdx : 해당 word가 주어진 문장에서 끝나는 위치
        Output : Array of dictionary
        -- lemma : 표제어
        -- sensid : 어깨번호
        -- definition : 정의 
        '''
        return []

    def get_def_candidate_list(self, word):
        '''
        주어진 word의 후보가 될 수 있는 definition list를 가져온다.
        기본적으로 정확히 일치하는 것만 가져온다.
        '''
        return data_util.get_corenet_matching_def_list(word)


    def get_word_origin_form(self, input):
        '''
        주어진 입력 텍스트 속의 단어를 CoreNet에서 사용하는 형식에 맞추어 반환하다.
        e.g) '명사' -> 그대로, '동사,형용사' -> 원형+'다'(예: 멋있다, 이루어지다)
        '''
        nlp_result = data_util.get_nlp_test_result(input['text'])
        if (nlp_result == None):
            return input['word']

        word_count = 0
        begin_byteIdx = 0
        for character in input['text']:
            if (word_count == input['beginIdx']):
                break
            word_count += 1
            begin_byteIdx += data_util.get_text_length_in_byte(character)

        morphs = nlp_result['sentence'][0]['WSD']
        for morp in morphs:
            if (morp['position'] == begin_byteIdx):
                # 동사 or 형용사 일 경우 wordnet 포맷에 맞추어 원형+'다' 형태로 반환한다. e.g.) '멋있' + '다'
                if (morp['type'] == 'VA' or morp['type'] == 'VV'):
                    return morp['text'] + '다'

        return input['word']


class BaselineDisambiguater(Disambiguater):
    '''
    Baseline Disambiguater. TF-IDF를 활용한다.
    '''
    def disambiguate(self, input):
        if not DataManager.isInitialized:
            return []

        input['word'] = self.get_word_origin_form(input)
        matching_def_list = data_util.get_real_corenet_matching_def_list(input['word'])
        max_cos_similiarity =  -1 * math.inf
        max_word_def = None

        for cornet_def in matching_def_list:
            if (len(cornet_def['definition1']) < 1 and len(cornet_def['usuage']) < 1):
                continue
            input_text = input['text']
            corenet_def_sent = data_util.convert_def_to_sentence(cornet_def)
            sentences = [input_text, corenet_def_sent]
            vec = DataManager.tfidf_obj.transform(sentences)
            cos_similarity = cosine_similarity(vec)[0][1]

            if ( cos_similarity > max_cos_similiarity ):
                max_cos_similiarity, max_word_def = cos_similarity, cornet_def

        if (max_word_def == None):
            return []
        return [{
            'lemma' : input['word'],
            'sensid' : '(' + str(max_word_def['vocnum']) + ',' + str(max_word_def['semnum']) + ')',
            'kortermnum' : max_word_def['kortermnum'],
            'definition' : max_word_def['definition1']
        }]

class KortermDisambiguater(Disambiguater):
    '''
    Kortermnum Disambiguater
    '''
    def disambiguate(self, input):
        if not DataManager.isInitialized:
            return []

        input['word'] = self.get_word_origin_form(input)
        matching_def_list = data_util.get_real_corenet_matching_def_list(input['word'])
        max_cos_similiarity =  -1 * math.inf
        max_word_def = None

        vocnum_list = set()

        input_vector = DataManager.tfidf_obj.transform([input['text']])
        for cornet_def in matching_def_list:
            vocnum_list.add(cornet_def['vocnum'])
            korterm = cornet_def['kortermnum']
            if (korterm not in DataManager.korenet_tfidf):
                continue
            if (type(korterm) is float or len(korterm) < 1):
                continue
            if (max_word_def is not None and max_word_def['kortermnum'] == korterm):
                continue

            korterm_vec = DataManager.korenet_tfidf[korterm]
            cos_similarity = cosine_similarity(input_vector, korterm_vec)[0][0]

            if ( cos_similarity > max_cos_similiarity ):
                max_cos_similiarity, max_word_def = cos_similarity, cornet_def

        if (max_word_def == None):
            return []

        return [{
            'lemma' : input['word'],
            'sensid' : '(' + str(max_word_def['vocnum']) + ',' + str(max_word_def['semnum']) + ')',
            'kortermnum' : max_word_def['kortermnum'],
            'definition' : max_word_def['definition1'],
            'confidence' : max_cos_similiarity,
            'candidate_num' : len(vocnum_list)
        }]


class RandomDisambiguater(Disambiguater):
    def disambiguate(self, input):
        if not DataManager.isInitialized:
            return []

        input['word'] = self.get_word_origin_form(input)
        matching_def_list = data_util.get_real_corenet_matching_def_list(input['word'])
        vocnum_set = set()
        for corenet_def in matching_def_list:
            vocnum_set.add(corenet_def['vocnum'])

        if (len(matching_def_list) < 1):
            return []


        random_value = random.randrange(0,len(matching_def_list))

        selected_def = matching_def_list[random_value]

        return [{
            'lemma': selected_def['term'],
            'sensid': '(' + str(selected_def['vocnum']) + ',' + str(selected_def['semnum']) + ')',
            'definition': selected_def['definition1'],
            'kortermnum': selected_def['kortermnum'],
            'candidate_num' : len(vocnum_set)
        }]

class HighFreqDisambiguater(Disambiguater):
    def disambiguate(self, input, corenet_lemma_obj):
        if not DataManager.isInitialized:
            return []

        input['word'] = self.get_word_origin_form(input)
        if (input['word'] not in corenet_lemma_obj):
            return []

        max_freq = -1
        max_candidate = None
        for candidate in corenet_lemma_obj[input['word']]:
            if (candidate['frequency'] > max_freq):
                max_freq = candidate['frequency']
                max_candidate = candidate


        return [{
            'lemma': input['word'],
            'sensid': '()',
            'definition': '',
            'kortermnum': list(max_candidate['korterm_set'])[0],
            'num_candidates': len(corenet_lemma_obj[input['word']])
        }]

class DemoMRFDisambiguater(Disambiguater):
    def initMRFDisambiguater(self):
        self.mrf_disambiguater = mrf_word_sense_disambiguation.MRFWordSenseDisambiguation()
        self.mrf_disambiguater.init()

    def disambiguate(self, input):
        text = input['text']
        nlp_result = data_util.get_nlp_test_result(text)
        if (nlp_result == None):
            return {'wsd_result':[]}

        st_time = int(time.time() * 1000)
        nlp_result = nlp_result['sentence']
        final_output_ary = []
        for sent_nlp_result in nlp_result:
            parse_for_argument = {}
            parse_for_argument['sentence'] = [sent_nlp_result]
            parse_result = self.mrf_disambiguater.disambiguate({'text':sent_nlp_result['text'], 'beginIdx':0, 'endIdx':0, 'word':''}, mode='ALL_WORD', parse_result=parse_for_argument)[2]
            output_ary = []
            for wsd_word_korterm in parse_result:
                wsd_word = wsd_word_korterm[0]
                wsd_korterm = wsd_word_korterm[1]
                candidate_list = self.mrf_disambiguater.corenet_lemma_obj[wsd_word]

                # 해당 Korterm이 들어있는 후보 번호 찾기
                selected_num = 0
                idx = 0
                for candidate in candidate_list:
                    for korterm in candidate['korterm_set']:
                        if wsd_korterm == korterm:
                            selected_num = idx
                            break
                    idx += 1

                # 선택된 후보에 definition이 하나도 없으면 무시
                if (len(candidate_list[selected_num]['definition_set']) < 1):
                    continue

                #korterm_set , sense_num_set
                matching_def_list = DataManager.corenet_data[wsd_word]
                one_word_ary_selected = []
                one_word_ary_cand = []
                for word_def in matching_def_list:
                    if len(word_def['definition1']) < 1: # 정의문이 없으면 무시
                        continue
                    try:
                        wordnet = corenet.getWordnet(wsd_word, float(word_def['vocnum']), float(word_def['semnum']), only_synonym=True)
                    except:
                        wordnet = []

                    en_synset = wordnet[0]['synset']._name if len(wordnet) > 0 else ''
                    en_lemmas = wordnet[0]['lemmas'] if len(wordnet) > 0 else []
                    en_definition = wordnet[0]['definition'] if len(wordnet) > 0 else ''

                    curr_sense_id = '(' + str(word_def['vocnum']) + ',' + str(word_def['semnum']) + ')'
                    item = {
                            'lemma' : wsd_word,
                            'senseid' : curr_sense_id,
                            'definition' : word_def['definition1'],
                            'usuage' : word_def['usuage'],
                            'beginIdx' : 0,
                            'endIdx' : 0,
                            'en_synset': en_synset,
                            'en_lemmas' : en_lemmas,
                            'en_definition' : en_definition,
                        }

                    is_selected_sense = False
                    for sense_id in candidate_list[selected_num]['sense_num_set']:
                        if (curr_sense_id == sense_id):
                            is_selected_sense = True
                            break

                    if (is_selected_sense):
                        item['score'] = 1.0
                        if (len(one_word_ary_selected) < 5):
                            one_word_ary_selected.append(item)
                    else:
                        item['score'] = 0.0
                        one_word_ary_cand.append(item)

                one_word_ary = one_word_ary_selected + one_word_ary_cand
                if (len(one_word_ary) > 0):
                    output_ary.append(one_word_ary)
            if(len(output_ary) > 0):
                final_output_ary.append(output_ary)
        print('time_elapsed %d' % (int(round(time.time() * 1000)) - st_time))
        return {'wsd_result': final_output_ary}


class DemoDisambiguater(Disambiguater):
    '''
    데모용 Disambiguater. 문장 하나만 입력으로 받고 문장 내의 모든 일반 명사에 대해서 WSD를 한다.
    '''
    def disambiguate(self, input):
        get_def_list_time = 0
        tfidf_time = 0
        get_wordnet_time = 0
        ttime = int(round(time.time() * 1000))
        if not DataManager.isInitialized:
            return []

        text = input['text']
        nlp_result = data_util.get_nlp_test_result(text)
        if (nlp_result == None):
            return {'wsd_result':[]}

        nlp_result = nlp_result['sentence']
        input_vector = DataManager.tfidf_obj.transform([text])
        final_output_ary = []
        for sent_nlp_result in nlp_result:
            morp_list = sent_nlp_result['WSD']
            output_ary = []
            for morp in morp_list:
                word = morp['text']
                if (morp['type'] == 'NNG'):

                    st_time = int(time.time()*1000)
                    matching_def_list = data_util.get_hanwoo_dic_matching_def_list(word)
                    get_def_list_time += (int(time.time()*1000) - st_time)
                    if (len(matching_def_list) < 1):
                        continue

                    st_time = int(time.time() * 1000)
                    for i in range(len(matching_def_list)):
                        cornet_def = matching_def_list[i]
                        if (len(cornet_def['definition1']) < 1):
                            cornet_def['cos_similarity'] = 0.0
                            continue
                        if (i > 0):
                            is_duplicate = False
                            for j in range(i):
                                prev_def = matching_def_list[j]
                                if ((prev_def['vocnum'] == cornet_def['vocnum'] and prev_def['semnum'] == cornet_def['semnum']) or prev_def['definition1'] == cornet_def['definition1']):
                                    is_duplicate = True
                                    break
                            if (is_duplicate):
                                cornet_def['cos_similarity'] = 0.0
                                continue

                        cornet_def_sent = data_util.convert_def_to_sentence(cornet_def)
                        sentences = [cornet_def_sent]
                        def_sent_vector = DataManager.tfidf_obj.transform(sentences)
                        cos_similarity = cosine_similarity(input_vector, def_sent_vector)[0][0]
                        cornet_def['cos_similarity'] = cos_similarity
                    tfidf_time += (int(time.time() * 1000) - st_time)

                    # beginIdx 구하기
                    chr_cnt = 0
                    position_cnt = 0
                    beginIdx = 0
                    for chr in text:
                        if (position_cnt == morp['position']):
                            beginIdx = chr_cnt
                            break
                        chr_cnt += 1
                        position_cnt += data_util.get_text_length_in_byte(chr)
                    endIdx = beginIdx + len(morp['text']) - 1

                    matching_def_list = sorted(matching_def_list, key=operator.itemgetter('cos_similarity'), reverse=True)
                    one_word_ary = []
                    st_time = int(time.time() * 1000)
                    for i in range(len(matching_def_list)):
                        word_def = matching_def_list[i]
                        if (word_def['cos_similarity'] < 0.0001):
                            break
                        try:
                            wordnet = corenet.getWordnet(word, float(word_def['vocnum']), float(word_def['semnum']), only_synonym=True)
                        except:
                            wordnet = []

                        en_synset = wordnet[0]['synset']._name if len(wordnet) > 0 else ''
                        en_lemmas = wordnet[0]['lemmas'] if len(wordnet) > 0 else []
                        en_definition = wordnet[0]['definition'] if len(wordnet) > 0 else ''

                        one_word_ary.append({
                            'lemma' : word_def['term'],
                            'senseid' : '(' + str(word_def['vocnum']) + ',' + str(word_def['semnum']) + ')',
                            'definition' : word_def['definition1'],
                            'usuage' : word_def['usuage'],
                            'beginIdx' : beginIdx,
                            'endIdx' : endIdx,
                            'score' : word_def['cos_similarity'],
                            'en_synset': en_synset,
                            'en_lemmas' : en_lemmas,
                            'en_definition' : en_definition,
                        })
                    elapsed = (int(time.time() * 1000) - st_time)
                    print ('--------------wordnet : %d'%(elapsed))
                    get_wordnet_time += elapsed
                    if (len(one_word_ary) > 0):
                        output_ary.append(one_word_ary)
            if (len(output_ary) > 0):
                final_output_ary.append(output_ary)
        print ('get_def_list_time %d'%get_def_list_time)
        print('tfidf_time %d' % tfidf_time)
        print('wordnet_time %d' % get_wordnet_time)
        print ('time_elapsed %d'%(int(round(time.time()*1000)) - ttime))
        return {'wsd_result' : final_output_ary}

class RESentenceDisambiguater(Disambiguater):

    def get_corenet_num(self, input_vector, word):
        korterm_list = []
        ttt = DataManager.corenet_obj[word]
        for idx in range(len(ttt)):
            item = ttt[idx]
            list = item['korterm_set']
            for korterm in list:
                korterm_list.append({'korterm': korterm, 'idx': idx})

        max_cos_similiarity = -10000.0
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

    def disambiguate(self, input):
        if ('text' not in input):
            return {'error':'Wrong JSON Format'}
        text = input['text'].strip()
        threshold = input['threshold'] if 'threshold' in input else 0.14
        etri_result = data_util.get_nlp_test_result(text)
        if (etri_result is None):
            return {'result':''}
        sent = etri_result['sentence'][0]

        match_korterm_list = []
        # sent_id 찾기
        text = sent['text'].strip()
        wsd_list = sent['WSD']

        input_vector = DataManager.tfidf_obj.transform([text])
        if (len(input_vector.data) == 0):
            return {'result':''}

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

                if (word not in DataManager.corenet_obj):
                    continue
                corenet_list = DataManager.corenet_obj[word]
                if (len(corenet_list) < 1):
                    continue

                wsd_result, confidence = self.get_corenet_num(input_vector, word)
                if (len(wsd_result) < 1):
                    continue
                if (confidence >= 0.14 or len(corenet_list) == 1):
                    word = word + '-@-' + str(wsd_result)
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

        return {'result':new_sent}

if __name__ == "__main__":
    DataManager.init_data()
    m_disambiguater = DemoMRFDisambiguater()
    m_disambiguater.initMRFDisambiguater()

    result = m_disambiguater.disambiguate({
        'text' : '애플은 스티브 잡스와 스티브 워즈니악과 론 웨인이 1976년에 설립한 컴퓨터 회사이다. 이전 명칭은 애플 컴퓨터였다. 최초의 개인용 컴퓨터 중 하나이며, 최초로 키보드와 모니터를 가지고 있는 애플 I을 출시하였고, 애플 II는 공전의 히트작이 되어 개인용 컴퓨터의 시대를 열었다.',
    })
