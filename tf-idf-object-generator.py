import random
import data_util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib

def construct_tf_idf(defintion_list):
    sent_list = data_util.convert_deflist_to_sent_list(defintion_list)
    tfidf = TfidfVectorizer(tokenizer=data_util.hannanum_tokenizer, min_df=3, max_df=20000)
    tfidf.fit(sent_list)
    joblib.dump(tfidf, './data/trained_tfidf.pkl')

def main():
    definition_list = data_util.read_corenet_definition_data()
    construct_tf_idf(definition_list)
    print('finished')

if __name__ == "__main__":
    main()