from __future__ import division
import numpy
from collections import Counter
import numpy

def read_data(filename):
    with open(filename) as f:
        pos = [l.strip().split()[:2] for l in f]
    pos = [['<s>','O'] if len(l)==0 else l for l in pos] 
    pos.insert(0, ['<s>','O'])
    pos.append(['<s>','O'])
    tags = set([tag for word,tag in pos])
    words = set([word for word,tag in pos])
    #pos_frame = [(pos[i-1][0],pos[i-1][1],pos[i][0],pos[i][1],pos[i+1][0]) for i in xrange(1,len(pos)-1) if pos[i][0]!='<s>']
    features1 = [(pos[i][0],pos[i][1]) for i in xrange(1,len(pos)-1) if pos[i][0]!='<s>']
    features2 = [(pos[i-1][1],pos[i][1]) for i in xrange(1,len(pos)-1) if pos[i][0]!='<s>']
    features = features1 + features2
    contexts = [feature[:-1] for feature in features]
    features_ins_num = len(features1)
    features_set = set(features)
    features_list = list(features_set)
    weights = numpy.zeros(len(features_list)) 
    features_count = Counter(features)
    contexts_count = Counter(contexts)
    e_ref = [features_count[feature]/features_ins_num for feature in features_list] 

    # test
    #print p_tag_given_context(('the','DT'), weights, features_list, features_set)
    #test(weights, features_list, features_set, tags)

    e_q = \
            [p_tag_given_context(features_list[i],weights,features_list,features_set)*contexts_count[features_list[i][:-1]]/features_ins_num
            for i in xrange(0, len(features_list))]
    e_ref = numpy.array(e_ref)
    e_q = numpy.array(e_q)
    temp = numpy.log(e_ref / e_q)
    weights = weights + temp

    #print tag_given_context('Rockwell',weights, features_list, features_set, tags)
    test(weights, features_list, features_set, tags)

    e_q = \
            [p_tag_given_context(features_list[i],weights,features_list,features_set)*contexts_count[features_list[i][:-1]]/features_ins_num
            for i in xrange(0, len(features_list))]
    e_ref = numpy.array(e_ref)
    e_q = numpy.array(e_q)
    temp = numpy.log(e_ref / e_q)
    weights = weights + temp

    #print tag_given_context('Rockwell',weights, features_list, features_set, tags)
    test(weights, features_list, features_set, tags)

def normalize(weights, features_list, current_feature):
    norm = sum(numpy.exp(weights[i]) for i in xrange(0,len(features_list)) if features_list[i][0]==current_feature[0])
    return norm

def p_tag_given_context(current_feature, weights, features_list, features_set):
    if not current_feature in features_set:
        return 0 
    index = features_list.index(current_feature)
    return numpy.exp(weights[index]) / normalize(weights, features_list, current_feature) 

def tag_given_context(context, weights, features_list, features_set, tags):
    tags = list(tags)
    ps = [p_tag_given_context((context,tag),weights,features_list,features_set) for tag in tags]
    return tags[ps.index(max(ps))]


def test(weights, features_list, features_set, tags):
    with open('test.txt') as f:
        pos_test = [l.strip().split()[:2] for l in f]
    pos_test = [['<s>','O'] if len(l)==0 else l for l in pos_test] 
    pos_test.insert(0, ['<s>','O'])
    pos_test.append(['<s>','O'])
    tags_test = set([tag for word,tag in pos_test])
    words_test = set([word for word,tag in pos_test])
    features_test = [(pos_test[i][0],pos_test[i][1]) for i in xrange(1,len(pos_test)-1) if pos_test[i][0]!='<s>']
    contexts_test = [feature[0] for feature in features_test]
    predicted_tags = [tag_given_context(context,weights,features_list,features_set,tags) for context in contexts_test]
    true_tags = [feature[1] for feature in features_test]
    correct_tags = [p for p,t in zip(predicted_tags,true_tags) if p==t]
    performance = len(correct_tags) / len(predicted_tags)
    print performance


if __name__ == '__main__':
    read_data('train.txt')

