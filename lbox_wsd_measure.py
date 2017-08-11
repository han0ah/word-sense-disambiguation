import json
import time
import os

def main():

    root_path = 'C:\SWRC_DATA\output_data_small\\'
    done_cnt = 0
    sttime = int(time.time())

    target_count = 0
    done_count = 0

    files = os.listdir(root_path)

    for filename in files:
        input_filepath = root_path + filename
        f = open(input_filepath, 'r', encoding='utf-8')

        for line in f:
            etri_obj = json.loads(line)
            sentence = etri_obj['sentence']
            for sent in sentence:
                wsd_list = sent['WSD']
                for wsd in wsd_list:
                    if (wsd['type'] == 'NNG' or wsd['type'] == 'VA' or wsd['type'] == 'VV'):
                        target_count += 1
                        if (wsd['corenet_voc_num'] > -1):
                            done_count += 1

            done_cnt += 1
            if (done_cnt%100 == 0):
                elapsed = int(time.time()) - sttime
                print("%d done. %d sec elapsed"%(done_cnt, elapsed))
                sttime = int(time.time())

        f.close()
    print ('total : %d , done : %d'%(target_count, done_count))


if __name__ == "__main__":
    main()

