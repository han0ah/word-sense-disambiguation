'''
REST API 처리를 담당하는 모듈 
'''
import json
from disambiguater_wrapper import DisambiguaterWrapper
from bottle import Bottle, run, post, request
from data_manager import DataManager

@post('/disambiguate')
def disambiguate():
    '''
    문장과 문장 내의 특정 단어가 주어졌을 때, 
    이 단어의 문맥 상에서의 의미와 일치하는 CoreNet 상의 표제어 및 어깨번호(의미번호) 를 반환하는 API
    :Sample Input Json : {“text”: “...”, “word”: “...”, “beginIdx”: “...”, “endIdx”: “...”}
    :Smpale Output Json : [{“lemma”: “...”, “senseid”: “...”, “definition”: “...”}, ... ]
    '''
    if not request.content_type.startswith('application/json'):
        return '{"error":"Content-type:application/json is required."}'

    request_str = request.body.read()
    try:
        request_str = request_str.decode('utf-8')
        input_obj = json.loads(request_str)
    except:
        return '{"error":"Failed to decode request text"}'

    m_disambiguater = DisambiguaterWrapper()
    result = m_disambiguater.disambiguate(input_obj)

    return json.dumps(result)

DataManager.init_data()
print(' data load finished ... ')
run(host='localhost', port=23000)


