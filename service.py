'''
REST API 처리를 담당하는 모듈 
'''
import json
from disambiguater import Disambiguater
from bottle import Bottle, run, post, request

@post('/disambiguate')
def test_api():
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
    except:
        return '{"error":"Failed to decode request text"}'

    test_obj = json.loads(request_str)

    disambiguate = Disambiguater()
    result = disambiguate.test(test_obj)

    return result

run(host='localhost', port=23000)


