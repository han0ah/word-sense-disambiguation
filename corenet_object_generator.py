import corenet
import pickle
from data_manager import DataManager
import data_util

def main():
    DataManager.init_data()
    print ('Data loaded')

    corenet_object = {}
    filepath = './data/corenet/koWord.dat'
    f = open(filepath, 'r', encoding='utf-8')
    count = 0
    for line in f:
        count += 1
        if (count < 16):
            continue

        if ((count-15)%200 == 0):
            print ("%d item finished"%(count-15))
        line = line.strip()
        if (len(line) < 1):
            continue

        items = line.split('\t')
        kortermnum = items[1]
        vocnum = int(items[2])
        semnum = int(items[3])
        word = items[8]

        if (word not in corenet_object):
            corenet_object[word] = []

        # vocnum, semnum 중복 제거
        isContinue = False
        for previtem in corenet_object[word]:
            if (previtem['vocnum'] == vocnum and previtem['semnum'] == semnum):
                isContinue = True
                break
        if (isContinue):
            continue

        definition, usuage = corenet.getDefinitionAndUsuage(word, vocnum, semnum)
        if (type(definition) == float or len(definition) < 1):
            definition = ""
        if (type(usuage) == float or len(usuage) < 1):
            usuage = ""

        text = word + '. ' \
               + definition.replace('～', word) + ' ' \
               + usuage.replace('～', word)
        vector_val = DataManager.tfidf_obj.transform([text])

        corenet_object[word].append({
            'vocnum':vocnum,
            'semnum':semnum,
            'definition1':definition,
            'usuage':usuage,
            'kortermnum':kortermnum,
            'vector':vector_val
        })


    f.close()
    pickle.dump(corenet_object, open('./data/corenet_obj.pickle','wb'))


if __name__ == "__main__":
    main()