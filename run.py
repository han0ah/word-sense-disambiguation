import disambiguater
from data_manager import DataManager

def main():
    DataManager.init_data()
    print ('Data loaded..')

    f_read = open('input.txt','r',encoding='utf-8')
    f_write = open('output.txt', 'w', encoding='utf-8')

    for line in f_read:
        line = line.strip()
        m_disambiguater = disambiguater.RESentenceDisambiguater()
        try:
            result = m_disambiguater.disambiguate({'text':line, 'threshold':0.05})
            f_write.write(result['result']+'\n')
        except:
            print ('error')
    f_read.close()
    f_write.close()
    print('Finished')

if __name__ == '__main__':
    main()
