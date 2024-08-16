import os
import config
import argparse
from Source.utils import load_file
from Source.processing import process

def pred(args):

    # load model file 
    model_file = os.path.join(config.output_path, f"{args.model_name}.pkl")
    
    # load vector file
    vect_file = os.path.join(config.output_path,f"{args.model_name}.pkl")

    vect =load_file(vect_file)
    
    # load model file
    model =load_file(model_file)

        # Tokenize the input text
    tokens = [process(args.text)]
    # Vectorize the tokens
    X = vect.transform(tokens)
    # Make predictions
    pred_prob = round(model.predict_proba(X)[0,1]*100, 2)
    print(f"Text: {args.text}")
    print(f"Probability of Positive Class: {pred_prob}")






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, help="Test review")
    parser.add_argument("--model_name",type=str,default="model_lr")
    args = parser.parse_args()
    pred(args)