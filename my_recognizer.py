import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
    #raise NotImplementedError
    #print(test_set.wordlist)

    test_sequences = list(test_set.get_all_Xlengths().values())
    counter = 0
    for X, length in  test_sequences:

        max_score = float("-Inf")
        chosen_word =""
        scores = dict()
        for word in models:
            model = models[word]
            try:
                model_score = model.score(X,length)
            except:
                model_score = float("-Inf")
                True
        #    print(model_score)
            scores[word] = model_score
            if model_score > max_score:
                max_score = model_score
                chosen_word = word
            #print(scores)

        probabilities.append(scores)
        guesses.append(chosen_word)
        counter = counter+1
        #print("Recognized word: {} True word: {}".format(chosen_word,test_set.wordlist[counter-1]))
    return probabilities, guesses
