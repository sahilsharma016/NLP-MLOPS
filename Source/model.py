from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from Source.utils import save_file,load_file

def vectorize(token_list,y, vect, min_df=5, ng_low=1,ng_high=3, test_size = 0.2, rs=42):
    
    # for count vector
    if vect == 'bow':
        vectorize = CountVectorizer(min_df=5)
    
    # for binary count ve
    elif vect == 'bowb':
        vectorize = CountVectorizer(binary=True,min_df=min_df)
    
    # n gram
    elif vect == 'ng':
        vectorize = CountVectorizer(min_df=min_df, ngram_range=(ng_low,ng_high))

    # tf-idf
        
    elif vect == 'tf':
        vectorize =TfidfVectorizer(min_df=min_df)
    
    else:
        raise Exception ("vect has to be 'bow','bowb','ng','tf' ")
    
    X = vectorize.fit_transform(token_list)

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,stratify=y,random_state=5)

    return X_train,X_test,y_train,y_test,vectorize