nlp_tokenize_dict = {}
dict_count = 0

def push_parse_result(parse_result):
    global dict_count,nlp_tokenize_dict

    dict_count += 1
    nlp_tokenize_dict[parse_result['text']] = {
        'parse_result' : parse_result,
        'index' : dict_count
    }

    if (dict_count >= 20):
        for key in nlp_tokenize_dict:
            if (nlp_tokenize_dict[key]['index'] == 1):
                del nlp_tokenize_dict[key]
            else:
                nlp_tokenize_dict[key]['index'] -=1
        dict_count -= 1







