import disambiguater
import data_manager

f = open('test_data2','r', encoding='utf-8')

def extract_disambiguate_obj_from_text(input_text):
    beginIdx = 0
    endIdx = 0
    word = ""
    count = 0
    bracket_opend = False
    for char in input_text:
        if char == "[":
            beginIdx = count
            bracket_opend = True
        elif char == "]":
            endIdx = count
            break
        else:
            count += 1
            if (bracket_opend):
                word = word + char
    text = input_text.replace('[','').replace(']','')
    return {
        'text' : text,
        'word' : word,
        'beginIdx' : beginIdx,
        'endIdx' : endIdx
    }

data_manager.DataManager.init_data()

total_hard_cnt = 0
ok_hard_cnt = 0
total_soft_cnt = 0
ok_soft_cnt = 0

idx = 0

for line in f:
    line = line.strip()
    if (len(line) < 1):
        continue

    idx += 1
    print(str(idx) + 'item finished')
    input_text, result, ishard = line.split('\t')
    ishard = ishard.strip()
    if (result.startswith('CONTEXT_')):
        continue

    if (ishard == '1'):
        total_hard_cnt += 1
    else:
        total_soft_cnt += 1

    input = extract_disambiguate_obj_from_text(input_text)
    sent = input['text']

    m_disambiguater = disambiguater.BaselineDisambiguater()
    system_output = m_disambiguater.disambiguate(input)
    if (len(system_output) == 0 and result.startswith('SENSE_')):
        if (ishard == '1'):
            ok_hard_cnt += 1
        else:
            ok_soft_cnt += 1
        print ('(-1,-1) match!')
    if (len(system_output) > 0 and system_output[0]['sensid'] == result):
        if (ishard == '1'):
            ok_hard_cnt += 1
        else:
            ok_soft_cnt += 1
        print (system_output[0]['sensid'] + ' ' + result + ' matched!')

print ('total_hard : %d / accurate : %d / accuracy : %.3f'%(total_hard_cnt, ok_hard_cnt, (ok_hard_cnt/total_hard_cnt)))
print ('total_soft : %d / accurate : %d / accuracy : %.3f'%(total_soft_cnt, ok_soft_cnt, (ok_soft_cnt/total_soft_cnt)))
f.close()

