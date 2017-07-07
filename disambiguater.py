import corenet
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
        matching_def_list = self.get_def_candidate_list(input['word'])

        max_cos_similiarity =  -1 * math.inf
        max_word_def = None

        for cornet_def in matching_def_list:
            if (len(cornet_def['definition1']) < 1 and len(cornet_def['usuage']) < 1):
                continue
            input_text = input['text']
            cornet_def_sent = data_util.convert_def_to_sentence(cornet_def)
            sentences = [input_text, cornet_def_sent]
            vec = DataManager.tfidf_obj.transform(sentences)
            cos_similarity = cosine_similarity(vec)[0][1]
            if ( cos_similarity > max_cos_similiarity ):
                max_cos_similiarity, max_word_def = cos_similarity, cornet_def

        if (max_word_def == None):
            return []
        return [{
            'lemma' : max_word_def['term'],
            'sensid' : '(' + str(max_word_def['vocnum']) + ',' + str(max_word_def['semnum']) + ')',
            'definition' : max_word_def['definition1']
        }]


class RandomDisambiguater(Disambiguater):
    def disambiguate(self, input):
        if not DataManager.isInitialized:
            return []

        input['word'] = self.get_word_origin_form(input)
        matching_def_list = self.get_def_candidate_list(input['word'])

        random_value = random.randrange(0,len(matching_def_list)+1)

        if (random_value == len(matching_def_list)):
            return []

        selected_def = matching_def_list[random_value]

        return [{
            'lemma': selected_def['term'],
            'sensid': '(' + str(selected_def['vocnum']) + ',' + str(selected_def['semnum']) + ')',
            'definition': selected_def['definition1']
        }]

class DemoDisambiguater(Disambiguater):
    '''
    데모용 Disambiguater. 문장 하나만 입력으로 받고 문장 내의 모든 일반 명사에 대해서 WSD를 한다.
    '''
    def disambiguate(self, input):
        ttime = int(round(time.time() * 1000))
        if not DataManager.isInitialized:
            return []

        text = input['text']
        nlp_result = data_util.get_nlp_test_result(text)
        if (nlp_result == None):
            return {'wsd_result':[]}

        nlp_result = nlp_result['sentence']

        final_output_ary = []
        for sent_nlp_result in nlp_result:
            morp_list = sent_nlp_result['WSD']
            output_ary = []
            for morp in morp_list:
                word = morp['text']
                if (morp['type'] == 'NNG'):

                    matching_def_list = self.get_def_candidate_list(word)
                    if (len(matching_def_list) < 1):
                        continue

                    for i in range(len(matching_def_list)):
                        cornet_def = matching_def_list[i]
                        if (len(cornet_def['definition1']) < 1):
                            cornet_def['cos_similarity'] = 0.0
                            continue
                        if (i > 0):
                            is_duplicate = False
                            for j in range(i):
                                prev_def = matching_def_list[j]
                                if (prev_def['vocnum'] == cornet_def['vocnum'] and prev_def['semnum'] == cornet_def['semnum']):
                                    is_duplicate = True
                                    break
                            if (is_duplicate):
                                cornet_def['cos_similarity'] = 0.0
                                continue
                        input_text = input['text']
                        cornet_def_sent = data_util.convert_def_to_sentence(cornet_def)
                        sentences = [input_text, cornet_def_sent]
                        vec = DataManager.tfidf_obj.transform(sentences)
                        cos_similarity = cosine_similarity(vec)[0][1]
                        cornet_def['cos_similarity'] = cos_similarity

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
                    for i in range(len(matching_def_list)):
                        word_def = matching_def_list[i]
                        if (word_def['cos_similarity'] < 0.0001):
                            break
                        try:
                            wordnet = corenet.getWordnet(word, float(word_def['vocnum']), float(word_def['semnum']))
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
                    output_ary.append(one_word_ary)
            final_output_ary.append(output_ary)
        print ('time_elapsed %d'%(int(round(time.time()*1000)) - ttime))
        return {'wsd_result' : final_output_ary}



if __name__ == "__main__":
    DataManager.init_data()
    m_disambiguater = DemoDisambiguater()
    result = m_disambiguater.disambiguate({
        'text' : '애플은 스티브 잡스와 스티브 워즈니악과 론 웨인이 1976년에 설립한 컴퓨터 회사이다. 이전 명칭은 애플 컴퓨터였다. 최초의 개인용 컴퓨터 중 하나이며, 최초로 키보드와 모니터를 가지고 있는 애플 I을 출시하였고, 애플 II는 공전의 히트작이 되어 개인용 컴퓨터의 시대를 열었다.'
    })
    print(result)