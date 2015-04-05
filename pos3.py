# MEM for POS
from __future__ import division
import numpy
from math import exp
from collections import Counter
import time

class MEM(object):
    """Meximum Entropy Model"""
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    def extract_features(self):
        """Extract features from training data"""
        # Construct different type of features
        # (w_i,t_i)
        f1 = [p for l in self.train_data for p in l]
        # (t_i-1,t_i)
        f2 = [(l[i-1][1],l[i][1]) for l in self.train_data for i in range(1,len(l))]

        # list of all features for computing empirical counts
        f_total = f1 + f2
        f_total_counter = Counter(f_total)

        # set of features for later use
        f_set = set(f_total)
        self.features_set = f_set

        # compute empirical counts
        self.features_list = list(f_set)
        self.features_num = len(self.features_list)
        self.emp_counts = numpy.zeros(self.features_num)
        for i in range(self.features_num):
            self.emp_counts[i] = f_total_counter[self.features_list[i]]

        #initialize parameter vector
        self.features_vec = numpy.zeros(self.features_num)
        self.tag_list = list(set([p[1] for p in f1]))

    def context2vec(self, context, tag, index):
        """Convert a given context and target tag at a position to a feature vector"""
        vec = numpy.zeros(self.features_num)
        # (w_i,t_i)
        if (context[index][0],tag) in self.features_set:
            vec[self.features_list.index((context[index][0],tag))] = 1
        # (t_i-1,t_i)
        if index > 0:
            if (context[index-1][1],tag) in self.features_set:
                vec[self.features_list.index((context[index-1][1],tag))] = 1
        return vec
        
    def tag_sentence(self, sentence):
        """Return a list of tuple, each is a pair of word and pos"""
        sentence = sentence.split()
        context = [(w,None) for w in sentence]
        for i in range(len(context)):
            scores = []
            for tag in self.tag_list:
                vec = self.context2vec(context, tag, i)
                scores.append(exp(numpy.dot(self.features_vec,vec)))
            scores = [score/sum(scores) for score in scores]
            context[i] = (context[i][0],self.tag_list[scores.index(max(scores))])
        return context

    def prob_tags(self, context, index):
        #vecs = [self.context2vec(context, tag, index) for tag in self.tag_list]
        #scores = [exp(numpy.dot(self.features_vec, vec)) for vec in vecs]
        scores = [exp(numpy.dot(self.features_vec,
            self.context2vec(context,tag,index))) for tag in self.tag_list]
        scores = [score/sum(scores) for score in scores]
        return scores

    def expected_counts(self):
        #exp_counts = numpy.zeros(self.features_num)
        exp_counts = sum(prob*self.context2vec(l,tag,i) for l in self.train_data for i
            in range(len(l)) for (tag,prob) in
            zip(self.tag_list,self.prob_tags(l, i)))

        #for l in self.train_data:
        #    for i in range(len(l)):
        #        probs = self.prob_tags(l, i)
        #        tags_probs = zip(self.tag_list, probs)
        #        exp_counts += sum([prob*self.context2vec(l, tag, i) for (tag, prob) in tags_probs])
        return exp_counts


    def gradient(self):
        grad1 = numpy.zeros(self.features_num)
        for l in self.train_data:
            for i in range(len(l)):
                grad1 += self.context2vec(l, l[i][1], i)
        print grad1

def read_data(pos_file):
    with open(pos_file) as f:
        pos = [tuple(l.strip().split()[:2]) for l in f]
    train_data = []
    temp = []
    for i in range(len(pos)):
        if len(pos[i]) > 0:
            temp.append(pos[i])
        else:
            train_data.append(temp)
            temp = []
    return train_data
        
if __name__ == '__main__':
    train_data = read_data('train.txt')
    test_data = read_data('test.txt')
    mem = MEM(train_data, test_data)
    mem.extract_features()
    #context = mem.tag_sentence('this is a cat')
    #mem.gradient()
    print mem.expected_counts()


