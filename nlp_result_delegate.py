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
        key_to_delete = None
        for key in nlp_tokenize_dict:
            if (nlp_tokenize_dict[key]['index'] == 1):
                key_to_delete = key
            else:
                nlp_tokenize_dict[key]['index'] -=1
        if key_to_delete is not None:
            del nlp_tokenize_dict[key_to_delete]
            dict_count -= 1







