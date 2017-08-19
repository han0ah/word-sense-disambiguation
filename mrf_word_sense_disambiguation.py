import pickle
import data_util
from pgmpy.models import MarkovModel
from pgmpy.factors import Factor
from pgmpy.inference import BeliefPropagation, VariableElimination


class MRFWordSenseDisambiguation:
    corenet_lemma_obj = None
    korterm_shortest_path = None

    def init(self):
        self.corenet_lemma_obj = pickle.load(open('./data/corenet_lemma_info_obj_with_freq.pickle', 'rb'))
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

    def disambiguate(self, input):
        text = input['text']


        etri_parser_result = data_util.get_nlp_test_result(text)
        if (etri_parser_result == None):
            return []
        input_word, input_beginIdx = self.get_word_origin_form(input, etri_parser_result)

        etri_parser_result = etri_parser_result['sentence'][0]

        X = []
        wsd_list = etri_parser_result['WSD']
        phrase_list = etri_parser_result['word']
        dependency_list = etri_parser_result['dependency']
        for wsd in wsd_list:
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

        # 각 X값 별로 Y값 구성
        Y = []
        for x in X:
            Yi = []
            y_list = self.corenet_lemma_obj[x['lemma']]
            for y_val in y_list:
#                korterm = list(y_val['korterm_set'])[0]
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
            for yval in Y[i]:
                values.append((yval['frequency']+1)*0.0001)
            node_factor = Factor([str(i)], cardinality=[len(Y[i])], values=values)
            model.add_factors(node_factor)

        for edge in markov_edges:
            values = []
            for yval1 in Y[int(edge[0])]:
                for yval2 in Y[int(edge[1])]:
                    if (yval1['kortermnum'] == yval2['kortermnum']):
                        shortest_path = 1
                    else:
                        shortest_path = self.korterm_shortest_path[yval1['kortermnum']][yval2['kortermnum']] + 1

                    values.append(1/shortest_path)
            edge_factor = Factor([edge[0],edge[1]], cardinality=[len(Y[int(edge[0])]),len(Y[int(edge[1])])], values=values)
            model.add_factors(edge_factor)

        inferrer = VariableElimination(model)
        result = inferrer.map_query()
        print(result)

        result_korterm = ""
        for i in range(len(X)):
            #TODO input idx 고려
            if str(i) in result:
                if 'is_target' in X[i]:
                    result_korterm = Y[i][result[str(i)]]['kortermnum']
                    break

        return [{'sensid':'(20,20)', 'kortermnum':result_korterm}]


if __name__ == '__main__':
    input = {
        'text' : '나는 어제 배를 먹었다.',
        'word' : '먹었',
        'beginIdx' : 9,
        'endIdx' : 10
    }

    disambiguater = MRFWordSenseDisambiguation()
    disambiguater.init()
    result = disambiguater.disambiguate(input)
    print(result)

debug = 1