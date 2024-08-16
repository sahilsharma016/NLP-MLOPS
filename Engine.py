import os
import config
import argparse
import pandas as pd
from Source.utils import save_file
from Source.model import vectorize
from Source.processing import process

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

def train_model(X_train,X_test,y_train,y_test):

    model = LogisticRegression()
    model.fit(X_train,y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # train accuracy
    train_acc= round(accuracy_score(y_train,train_pred)*100,2)

    # test_accuracy
    
    test_acc = round(accuracy_score(y_test,test_pred)*100,2)

    print(f"t=Train accuracy:  {train_acc}" )
    print(f"Test accuracy: {test_acc}")
    return model



def main(args):
    # create path for input
    input_file = os.path.join(config.input_path,args.file_name)
    print(input_file)
    # path for vectorizer
    vect_file  = os.path.join(config.output_path,f"{args.output_name}.pkl")

    # model path
    model_file = os.path.join(config.output_path,f"{args.output_name}_lr.pkl")

    # read data
    data = pd.read_excel(input_file)

    data = data[[config.text_col,config.label_col]]
    reviews = list(data[config.text_col])

    # preprocess
    reviews = [process(r,config.stem) for r in reviews]

    y =data[config.label_col]

    # vectorize the data and split into test train
    X_train, X_test, y_train, y_test, vectorizer = vectorize(reviews, y,
                                                             vect=args.vectorizer,
                                                             min_df=config.min_df,
                                                             ng_low=config.ng_low,
                                                             ng_high=config.ng_high,
                                                             test_size=config.test_size,
                                                             rs=config.rs)
    # Save the vectorizer
    save_file(vect_file, vectorizer)
    # Train the model
    model = train_model(X_train, X_test, y_train, y_test)
    # Save the model file
    save_file(model_file, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str, default="Canva_reviews.xlsx",
                        help="Input file name")
    parser.add_argument("--vectorizer", type=str, default="bow",
                        help="Vectorizer, one of - 'bow', 'bowb', 'ng','tf'")
    parser.add_argument("--output_name", type=str, default="model",
                        help="Output file name")
    args = parser.parse_args()
    main(args)
