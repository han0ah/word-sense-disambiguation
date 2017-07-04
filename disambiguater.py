import math
import data_util
import random
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
        '''
        return []

class BaselineDisambiguater(Disambiguater):
    '''
    Baseline Disambiguater. TF-IDF를 활용한다.
    '''
    def disambiguate(self, input):
        if not DataManager.isInitialized:
            return []
        matching_def_list = self.get_def_candidate_list(input['word'])

        max_cos_similiarity =  -1 * math.inf
        max_word_def = None

        for cornet_def in matching_def_list:
            if (len(cornet_def['definition1']) < 1):
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

    def get_def_candidate_list(self, word):
        '''
        주어진 word의 후보가 될 수 있는 definition list를 가져온다.
        이 Baseline의 경우에는 term이 word랑 완전히 일치할 때만 반환.
        '''
        matching_def_list = []
        for word_def in DataManager.corenet_data:
            if (word_def['term'] == word):
                matching_def_list.append(word_def)
        return matching_def_list

class RandomDisambiguater(Disambiguater):
    def disambiguate(self, input):
        if not DataManager.isInitialized:
            return []
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

    def get_def_candidate_list(self, word):
        '''
        주어진 word의 후보가 될 수 있는 definition list를 가져온다.
        이 Disambiguater는 term이 word랑 완전히 일치할 때만 반환.
        '''
        matching_def_list = []
        for word_def in DataManager.corenet_data:
            if (word_def['term'] == word):
                matching_def_list.append(word_def)
        return matching_def_list

if __name__ == "__main__":
    DataManager.init_data()
    m_disambiguater = BaselineDisambiguater()
    m_disambiguater.disambiguate({
        'text' : '밤나무의 열매. 가시가 많이 난 송이에 싸여 있고 갈색 겉껍질 안에 얇고 맛이 떫은 속껍질이 있다.',
        'word' : '밤',
        'beginIdx' : 0,
        'endIdx': 1
    })