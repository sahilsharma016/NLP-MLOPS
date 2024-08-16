# All the processing required for stopwords + special char.

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,LancasterStemmer
from nltk.tokenize import word_tokenize,RegexpTokenizer

sw = stopwords.words('english')

# this will remove any special words (except digits)
tokenizer = RegexpTokenizer(r'\w+')




def process(review,stem ='p'):
    review = review.lower()

    tokens = word_tokenize(review)

    # remove stopwords
    tokens = [t for t in tokens if t not in sw]

    # remove punctuation
    tokens = [tokenizer.tokenize(t) for t in tokens]
    tokens = [t for t in tokens if len(t)>0]
    tokens = ["".join(t) for t in tokens]

    # create stemmmer 

    if stem == 'p':
        stemmer = PorterStemmer()
    elif stem =='l':
        stemmer = LancasterStemmer()
    else:
        raise Exception ("stem should be either 'p' or 'l' ")
    
    # stemming
    tokens = [stemmer.stem(t) for t in tokens] 
    return " ".join(tokens)