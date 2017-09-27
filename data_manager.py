import pickle
from sklearn.externals import joblib

class DataManager():
    # train된 tfidf 오브젝트
    tfidf_obj = None

    # corenet definition data
    corenet_data = None

    # corenet object data
    corenet_obj = None

    # korterm tfidf object
    korenet_tfidf = None

    isInitialized = False

    @staticmethod
    def init_data():
        DataManager.tfidf_obj = joblib.load('./data/trained_tfidf_etri_tokenize.pkl')
        DataManager.corenet_obj = pickle.load(open('./data/corenet_lemma_info_obj_new.pickle', 'rb'))
        #DataManager.corenet_data = data_util.read_corenet_definition_data()
        DataManager.korenet_tfidf = joblib.load('./data/korterm_tfidf.pickle')
        DataManager.isInitialized = True
