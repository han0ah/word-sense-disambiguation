import pickle
import data_util
import math
from pgmpy.models import MarkovModel
import pgmpy.factors
from pgmpy.inference import BeliefPropagation, VariableElimination, Mplp

class MRFWordSenseDisambiguation:
    corenet_lemma_obj = None
    korterm_shortest_path = None
    korterm_cooccur_freq = None

    def init(self):
        self.corenet_lemma_obj = pickle.load(open('./data/corenet_lemma_info_obj_with_freq_014_final.pickle', 'rb'))
        self.korterm_cooccur_freq = pickle.load(open('./data/korterm_cooccur_freq_014_final.pickle', 'rb'))
        self.korterm_shortest_path = pickle.load(open('./data/korterm_shortest_path.pickle', 'rb'))


    def get_word_origin_form(self, input, nlp_result):
        '''
        주어진 입력 텍스트 속의 단어를 CoreNet에서 사용하는 형식에 맞추어 반환하다.
        e.g) '명사' -> 그대로, '동사,형용사' -> 원형+'다'(예: 멋있다, 이루어지다)
        '''
        if (nlp_result == None):
            return input['word'],-1

        word_count = 0
        begin_byteIdx = 0
        for character in input['text']:
            if (word_count == input['beginIdx']):
                break
            word_count += 1
            begin_byteIdx += data_util.get_text_length_in_byte(character)

        morphs = nlp_result['sentence'][0]['WSD']
        begin_idx = -1
        word = input['word']
        for morp in morphs:
            if (morp['position'] == begin_byteIdx):
                begin_idx = morp['begin']
                word = morp['text']
                # 동사 or 형용사 일 경우 wordnet 포맷에 맞추어 원형+'다' 형태로 반환한다. e.g.) '멋있' + '다'
                if (morp['type'] == 'VA' or morp['type'] == 'VV'):
                    word = morp['text'] + '다'
                break

        return word, begin_idx

    def disambiguate(self, input, mode="ONE_WORD", parse_result=None):
        text = input['text']


        etri_parser_result = data_util.get_nlp_test_result(text) if parse_result == None else parse_result
        if (etri_parser_result == None):
            return []
        input_word, input_beginIdx = self.get_word_origin_form(input, etri_parser_result)

        etri_parser_result = etri_parser_result['sentence'][0]

        X = []
        wsd_list = etri_parser_result['WSD']
        phrase_list = etri_parser_result['word']
        dependency_list = etri_parser_result['dependency']
        iii = 0
        for wsd in wsd_list:
            iii += 1
            if (wsd['type'] == 'NNG' or wsd['type'] == 'VA' or wsd['type'] == 'VV'):
                t_word = wsd['text'] + ('다' if (wsd['type'] != 'NNG') else '')
            else:
                continue

            # dependency parsing 시 몇번째인지 구하는 것
            phrase_idx = 0
            for phrase in phrase_list:
                if (wsd['begin'] >= phrase['begin'] and wsd['end'] <= phrase['end']):
                    phrase_idx = phrase['id']
                    break

            if (t_word in self.corenet_lemma_obj and len(self.corenet_lemma_obj[t_word]) > 0):
                new_obj = {'lemma':t_word, 'idx':phrase_idx}
                if wsd['begin'] == input_beginIdx:
                    new_obj['is_target'] = True
                X.append(new_obj)
                etri_parser_result['WSD'][iii-1]['wsd_index'] = len(X)-1


        # 각 X값 별로 Y값 구성
        Y = []
        for x in X:
            Yi = []
            y_list = self.corenet_lemma_obj[x['lemma']]
            for y_val in y_list:
                for korterm in y_val['korterm_set']:
                   Yi.append({'kortermnum':korterm, 'frequency':y_val['frequency']})

            Y.append(Yi)

        '''
        iddx = 0
        for y in Y:
            print (X[iddx]['lemma'] + " : ", end='')
            for yval in y:
                print (yval['kortermnum'] + ', ', end='')
            print ('\n', end='')
            iddx += 1
        '''

        markov_edges = []
        for dependency in dependency_list:
            st = dependency['id']
            en = dependency['head']
            if (en == -1):
                continue

            for i in range(len(X)):
                for j in range(len(X)):
                    if (i == j):
                        continue
                    if (X[i]['idx'] == st and X[j]['idx'] == en):
                        markov_edges.append((str(i),str(j)))

        model = MarkovModel(markov_edges)
        for i in range(len(X)):
            model.add_node(str(i))

        for i in range(len(X)):
            values = []
            total = 0.0
            for yval in Y[i]:
                val = math.log((yval['frequency']+2.7183))
                values.append(val)
                total += val
            values = [v / total for v in values]
            node_factor = pgmpy.factors.Factor([str(i)], cardinality=[len(Y[i])], values=values)
            model.add_factors(node_factor)

        for edge in markov_edges:
            values = []
            total = 0.0
            for yval1 in Y[int(edge[0])]:
                for yval2 in Y[int(edge[1])]:
                    korterm1 = yval1['kortermnum']
                    korterm2 = yval2['kortermnum']
                    val = self.korterm_cooccur_freq[korterm1][korterm2]+1
                    total += val
                    values.append(val)

                    # Shortest Path 이용하는 방식
                    '''
                    if (yval1['kortermnum'] == yval2['kortermnum']):
                        shortest_path = 1
                    else:
                        shortest_path = self.korterm_shortest_path[yval1['kortermnum']][yval2['kortermnum']] + 1
                    
                    values.append(1/shortest_path)
                    total += (1/shortest_path)
                    '''

            values = [v/total for v in values]
            edge_factor = pgmpy.factors.Factor([edge[0],edge[1]], cardinality=[len(Y[int(edge[0])]),len(Y[int(edge[1])])], values=values)
            model.add_factors(edge_factor)

        inferrer = Mplp(model)
        inferrer.map_query()
        result = {}
        for key in inferrer.best_assignment.keys():
            refined_key = str(key).replace("frozenset({'",'').replace("'})",'')
            result[refined_key] = inferrer.best_assignment[key]

        if (mode == 'ONE_WORD'):
            result_korterm = ""
            for i in range(len(X)):
                #TODO input idx 고려
                if str(i) in result:
                    if 'is_target' in X[i]:
                        result_korterm = Y[i][result[str(i)]]['kortermnum']
                        break

            return [{'sensid':'(20,20)', 'kortermnum':result_korterm}]
        else:
            result2 = []
            for i in range(len(X)):
                result_korterm =  Y[i][result[str(i)]]['kortermnum']
                result[str(i)] = result_korterm
                result2.append((X[i]['lemma'],result_korterm))
            return result,etri_parser_result,result2


if __name__ == '__main__':
    input = {
        'text' : '사과가 참 맛있다.',
        'word' : '불리다',
        'mode' : 'ALL_WORD',
        'beginIdx' : 0,
        'endIdx' : 0
    }

    disambiguater = MRFWordSenseDisambiguation()
    disambiguater.init()
    result = disambiguater.disambiguate(input, mode='All_WORD')
    print(result)

debug = 1