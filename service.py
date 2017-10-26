'''
REST API 처리를 담당하는 모듈 
'''
import json
from disambiguater_wrapper import DisambiguaterWrapper
import disambiguater
from bottle import Bottle, run, post, request, response
from data_manager import DataManager


def enable_cors(fn):
    def _enable_cors(*args, **kwargs):
        # set CORS headers
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token'

        if request.method != 'OPTIONS':
            return fn(*args, **kwargs)
    return _enable_cors

@post('/disambiguate', method=['OPTIONS','POST'])
@enable_cors
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

    m_disambiguater = DataManager.mrf_disambiguater()
    result = m_disambiguater.disambiguate(input_obj)

    return json.dumps(result)


@post('/disambiguate_re', method=['OPTIONS','POST'])
@enable_cors
def disambiguate_re():
    '''
    CNN/PCNN Relation Extraction 모듈에서 사용하는 disambiguater
    문장을 주면 문장 속 단어에 Sense 번호를 붙여서 반환한다.
    :return: 
    '''
    if not request.content_type.startswith('application/json'):
        return '{"error":"Content-type:application/json is required."}'

    request_str = request.body.read()
    try:
        request_str = request_str.decode('utf-8')
        input_obj = json.loads(request_str)
    except:
        return '{"error":"Failed to decode request text"}'

    m_disambiguater = disambiguater.RESentenceDisambiguater()
    result = m_disambiguater.disambiguate(input_obj)

    return json.dumps(result)

DataManager.init_data()
print(' data load finished ... ')
run(host='localhost', port=23000)


