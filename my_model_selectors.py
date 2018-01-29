import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        best_score=float("+Inf")
        model_scores = []
        model_lengths = range(self.min_n_components, self.max_n_components)
        for num_states in model_lengths:
            try:
                model = self.base_model(num_states)
                score = model.score(self.X, self.lengths)

                # BIC = -2*logL + plogN
                # p - number of parameters
                # N - number of datapoints

                # number of parameters can be computed from
                p = num_states * num_states + 2*num_states*len(self.X[0])-1
                N = len(self.X)

                BIC = -2 * score + p*math.log(N)
                model_scores.append(BIC)

            except:
        #        print("Not training for model_length={}".format(num_states))
                model_scores.append(float("+Inf"))

        #print(model_scores)
        if(len(model_scores)):
            #print("best model {} ".format(np.argmax(model_scores)))
            return self.base_model(model_lengths[np.argmin(model_scores)])



class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        word_scores = dict()
        num_states=6
        model_scores = []
        model_lengths = range(self.min_n_components, self.max_n_components)
        for num_states in model_lengths:
            total_score = 0
            model = self.base_model(num_states)
            for word in self.words:
                X, lengths = self.hwords[word]
                try:
                    word_scores[word] = model.score(X, lengths)
                    if self.this_word != word:
                        total_score = total_score + word_scores[word]
                except:
                    word_scores[word] = float("-Inf")

            #print(word_scores)
            #print(word_scores[self.this_word])
            avg_score_all_words = total_score / (len(self.words)-1)
            delta_score = word_scores[self.this_word] - avg_score_all_words
            model_scores.append(delta_score)
        if(len(model_scores)):
            #print("best model {} ".format(np.argmax(model_scores)))
            return self.base_model(model_lengths[np.argmax(model_scores)])
        #print(avg_score_all_words)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        import asl_utils
        import numpy as np
        from sklearn.model_selection import KFold
        n_splits=3
        split_method = KFold()
        best_score=float("+Inf")
        model_scores = []
        model_lengths = range(self.min_n_components, self.max_n_components)

        for num_states in model_lengths:
            # Used to calculate the average LL across the folds
            number_of_splits = 0
            sum_of_scores = 0

            if(len(self.sequences) < n_splits):
                break

            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    number_of_splits = number_of_splits+1
                    #print("Train fold indices:{} Test fold indices:{}".format(cv_train_idx, cv_test_idx))  # view indices of the folds
                    #try:
                    X,lengths = combine_sequences(cv_train_idx, self.sequences)
                    train_x, train_lengths = combine_sequences(cv_test_idx,self.sequences)
                    self.X = X
                    self.lengths = lengths
                    model = self.base_model(num_states)
                    #print(train_x)
                    try:
                        score = model.score(train_x, train_lengths)
                    except:
                        score = 0
                    #print(score)
                    sum_of_scores = sum_of_scores + score
                    #except:
                    #    print("Model not training for {}".format(cv_train_idx))

            avg_score = sum_of_scores / number_of_splits

            model_scores.append( avg_score)

        #    print(avg_score)
        #    print(model_scores)


        #print(model_scores)
        if(len(model_scores)):
        #    print("best model {} ".format(np.argmax(model_scores)))
            return self.base_model(model_lengths[np.argmax(model_scores)])

        # TODO implement model selection using CV
        return self.base_model(3)
