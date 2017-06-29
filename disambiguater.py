import math
import data_util
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

class BaselineDisambiguater(Disambiguater):
    '''
    Baseline Disambiguater. TF-IDF를 활용한다.
    '''
    def disambiguate(self, input):
        if not DataManager.isInitialized:
            return []
        matching_def_list = self.get_matching_def_list(input['word'])

        max_cos_similiarity =  -1 * math.inf
        max_word_def = None

        for cornet_def in matching_def_list:
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
            'sensid' : max_word_def['vocnum'],
            'definition' : max_word_def['definition1']
        }]

    def get_matching_def_list(self, word):
        '''
        해당 word와 일치하는 term을 갖는 corenet 상의 데이터 목록을 반환한다. 
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