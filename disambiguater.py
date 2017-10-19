import math
import data_util
import json
import nlp_result_delegate
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
        etri_result = json.loads(input['etri_result']) if 'etri_result' in input else data_util.get_nlp_test_result(text)
        if (etri_result is None):
            return {'result':''}
        sent = etri_result['sentence'][0]
        nlp_result_delegate.push_parse_result(sent)

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
                if (confidence >= threshold or len(corenet_list) == 1):
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
    m_disambiguater = RESentenceDisambiguater()

    result = m_disambiguater.disambiguate({
        'text' : '사과는 맛있다.',
        'threshold' : 0.14,
        'etri_result' : '{"sentence":[{"id" : 0,"reserve_str" : "","text" : "사과는 맛있다.","morp" : [{"id" : 0, "lemma" : "사과", "type" : "NNG", "position" : 0, "weight" : 0.437589 },{"id" : 1, "lemma" : "는", "type" : "JX", "position" : 6, "weight" : 0.0287565 },{"id" : 2, "lemma" : "맛있", "type" : "VA", "position" : 10, "weight" : 0.9 },{"id" : 3, "lemma" : "다", "type" : "EF", "position" : 16, "weight" : 0.132573 },{"id" : 4, "lemma" : ".", "type" : "SF", "position" : 19, "weight" : 1 }],"WSD" : [{"id" : 0, "text" : "사과", "type" : "NNG", "scode" : "05", "weight" : 2.2, "position" : 0, "begin" : 0, "end" : 0},{"id" : 1, "text" : "는", "type" : "JX", "scode" : "00", "weight" : 1, "position" : 6, "begin" : 1, "end" : 1},{"id" : 2, "text" : "맛있", "type" : "VA", "scode" : "00", "weight" : 0, "position" : 10, "begin" : 2, "end" : 2},{"id" : 3, "text" : "다", "type" : "EF", "scode" : "00", "weight" : 1, "position" : 16, "begin" : 3, "end" : 3},{"id" : 4, "text" : ".", "type" : "SF", "scode" : "00", "weight" : 1, "position" : 19, "begin" : 4, "end" : 4}],"word" : [{"id" : 0, "text" : "사과는", "type" : "", "begin" : 0, "end" : 1},{"id" : 1, "text" : "맛있다.", "type" : "", "begin" : 2, "end" : 4}]}]}'
    })