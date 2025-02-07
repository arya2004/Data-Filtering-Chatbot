import pandas as pd
import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import sys

import warnings
warnings.filterwarnings('ignore')

if len(sys.argv)==2:
    print(sys.argv)
    submission_name = sys.argv[1]
    df = pd.read_excel(f"{submission_name}.xlsx")


nltk.download('punkt')
stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')
def cosine_sim(text1, text2):
    if text1==text2:
        return 1
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).toarray())[0,1]





def filter_table_process(name,maxlen):
    arr = []
    for ele in df[name].tolist():
        if isinstance(ele,str):
            temp = [int(x)+1 for x in ele.split(',')]
            arr.append(temp)
        else:
            arr.append(ele+1)

    temp = []
    ans_arr = []
    for ele in arr:
        if isinstance(ele,list):
            if len(ele)!=maxlen:
                temp = [0 for i in range(maxlen-len(ele))]
                temp.extend(ele)
                ans_arr.append(temp)
            else:
                ans_arr.append(ele)
        else:
            temp = [0 for i in range(maxlen-1)]
            temp.append(ele)
            ans_arr.append(temp)

    return ans_arr


def table_score_cal(row_maxlen=1000,column_maxlen=11):
    
    ans_row = filter_table_process(name='row index',maxlen=row_maxlen)
    pred_row = filter_table_process(name='filtered row index',maxlen=row_maxlen)

    new = [[ele for ele in range(1,row_maxlen+1)]]
    mlb_row = MultiLabelBinarizer()
    mlb_row.fit(new)

    ans_row = mlb_row.transform(ans_row)
    pred_row = mlb_row.transform(pred_row)
    row_score = f1_score(ans_row, pred_row, average='micro')


    ans_columns = filter_table_process(name='column index',maxlen=column_maxlen)
    pred_columns = filter_table_process(name='filtered column index',maxlen=column_maxlen)

    new = [[ele for ele in range(1,column_maxlen+1)]]
    mlb_column = MultiLabelBinarizer()
    mlb_column.fit(new)

    ans_columns = mlb_column.transform(ans_columns)
    pred_columns = mlb_column.transform(pred_columns)
    column_score = f1_score(ans_columns, pred_columns, average='micro')

    table_score = np.mean([row_score,column_score])
    return table_score


def generated_response_process():
    ans_str, ans_fl = dict(), dict()
    for idx,ele in df["answer"].items():
        try:
            float(ele)
            ans_fl[idx] = str(ele)
        except ValueError:
            # print("Not a float")
            ans_str[idx] = ele

    return ans_str,ans_fl


def generated_response_score():
    ans_str,ans_fl = generated_response_process()

    res_fl = [str(df['generated response'].iloc[idx]) for idx in ans_fl.keys()]
    num_res_score = f1_score(res_fl, list(ans_fl.values()), average='micro') if len(res_fl)!=0 else 0

    text_res_scores = [cosine_sim(str(ele),str(df["generated response"].iloc[idx])) for idx,ele in ans_str.items()]
    text_res_score = np.mean(text_res_scores)
    text_res_score = np.nan_to_num(text_res_score)

    res_score = np.mean([num_res_score,text_res_score])

    return res_score


def final_score(submission_name,row_maxlen=1000,column_maxlen=11):
    global df
    df = pd.read_excel(f"{submission_name}.xlsx")
    table_score = table_score_cal(row_maxlen=row_maxlen,column_maxlen=column_maxlen)
    res_score = generated_response_score()
    print("Table Filter Score :",table_score)
    print("Generated response Score :",res_score)
    total_score = np.mean([table_score,res_score])
    print("Total Score :",total_score)
    final_score = ((table_score*0.7) + (res_score*0.3))
    print("Final Weighted Score :",final_score)
    return table_score,res_score,total_score,final_score



if __name__ == "__main__":
    final_score(submission_name,row_maxlen=1000,column_maxlen=11)