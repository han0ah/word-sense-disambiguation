class Disambiguater:
    '''
    문장과 문장 내의 특정 단어가 주어졌을 때, 
    이 단어의 문맥 상에서의 의미와 일치하는 CoreNet 상의 표제어 및 어깨번호(의미번호) 를 반환하는 기능을 수행하는 모듈들의
    추상 클래스
    각 Disambiguater 모듈들은 이 모듈을 상속받아서 정의된 interface를 구현한다.
    '''
    def disambiguate(self):
        return []

    def test(self, input):
        return '{"result":"ok"}'


class BaselineDisambiguater(Disambiguater):
    '''
    Baseline Disambiguater TF-IDF를 활용한다.
    '''
    def disambiguate(self):
        return []