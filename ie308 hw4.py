#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 22:08:30 2018

@author: jsshenkman



• Which companies went bankrupt in month X of year Y?
o The answer should be the name of the companies.
• What affects GDP? What percentage of drop or increase is associated with this property?
o It should work in the following way: What affects GDP? Answer: unemployment,
interest rates, … Then the user asks the follow-up question: What percentage of drop or
increase is associated with Z? (Here Z can be: unemployment or interest rates or any
other property returned.) Answer: 1.0%
• Who is the CEO of company X?
o The answer should be the first and last name of the CEO. 

"""
import numpy as np
import pandas as pd
import pysolr
# import textmining
import os
#from lupyne import engine
import base64
from pickle import dumps, loads
import unittest
import tempfile
import nltk
import pandas as pd
import numpy as np
from nltk.chunk import conlltags2tree, tree2conlltags
from nltk import word_tokenize, pos_tag, ne_chunk,sent_tokenize
from nltk.stem import porter
import matplotlib
from collections import Counter
from nltk.corpus import names, stopwords
from nltk.util import ngrams
import random
import string
from nltk.data import load
from nltk.corpus import wordnet
import calendar
from nltk.stem.porter import *
import sys

# set up logging
old_stdout = sys.stdout

log_file = open("hw4_1.log","w")

sys.stdout = log_file



# get documents
documents_2014 = os.listdir('/Users/jsshenkman/Documents/python/2014')
documents_2014 = ['/Users/jsshenkman/Documents/python/2014/' + element for element in documents_2014]
documents_2013 = os.listdir('/Users/jsshenkman/Documents/python/2013')
documents_2013 = ['/Users/jsshenkman/Documents/python/2013/' + element for element in documents_2013]

documents = documents_2013+documents_2014


def get_data(filepath):
    ""
    ""
    with open (filepath, "r",errors='ignore') as myfile:
        data=myfile.read()
    # get rid of \ and \n and \r
    data = data.replace('\xa0',' ')
    data = data.replace('\n',' ')
    data = data.replace('\r',' ')
    data = data.replace('\'',"")
    return data

#data = get_data(documents[0])

# create document term matrix
#solr = pysolr.Solr('http://localhost:8983/solr/')
#solr.add([{'id': documents[0], 'title': data }])

def add_files(solr):
    """
    """
    for filepath in documents:
        data = get_data(filepath)
        solr.add([{'id': filepath, 'title': data }])

def add_files_stemmed(solr):
    """
    """
    for filepath in documents:
        data = get_data(filepath)
        data = word_tokenize(data)
        porter_stemmer = PorterStemmer()
        data = [porter_stemmer.stem(word) for word in data]
        solr.add([{'id': filepath, 'title': data }])

solr = pysolr.Solr('http://localhost:8983/solr/hw4_2',results_cls=dict)
# solr_stem = pysolr.Solr('http://localhost:2222/solr/hw4_3',results_cls=dict)

####### ADD FILES HERE
# add_files(solr)
# add_files_stemmed(solr_stem)
results = solr.search('title:Warren Buffet',fl='score')


# set stopwords
stopwords = set(stopwords.words('english'))



# Enron

q3 = 'What affects GDP? What percentage of drop or increase is associated with this property?'

def find_name_document(document):
    """
    """
    # initialize
    names = list()
    #sentences = sent_tokenize(document)
    #words = [word_tokenize(element) for element in sentences]
    words = word_tokenize(document)
    # get the correct tags for all the sentences
    #iob_tagged = [tree2conlltags(ne_chunk(pos_tag(element))) for element in words]
    #iob_tagged = tree2conlltags(ne_chunk(pos_tag(words)))
    tree = ne_chunk(pos_tag(words))
    # get the people trees
    people_trees = [element for element in tree if ((type(element)==nltk.tree.Tree) and (element.label() == 'PERSON') and (len(element) ==2))]
    people_names = [element[0][0]+' ' + element[1][0] for element in people_trees]
    people_names = np.unique(people_names)
    #### return all the people
    # get the ne labels
    #ne_labels = [element[2] for element in iob_tagged]
    # make bigrams
    return people_names

def find_company_document(document):
    """
    """
    companies = list()
    #sentences = sent_tokenize(document)
    #words = [word_tokenize(element) for element in sentences]
    words = word_tokenize(document)
    # get the correct tags for all the sentences
    #iob_tagged = [tree2conlltags(ne_chunk(pos_tag(element))) for element in words]
    #iob_tagged = tree2conlltags(ne_chunk(pos_tag(words)))
    tree = ne_chunk(pos_tag(words))
    # get company trees
    company_trees = [element for element in tree if ((type(element)==nltk.tree.Tree) and (element.label() == 'ORGANIZATION'))]
    # combine to take words
        
    company_words = [list(np.array(element)[:,0]) for element in company_trees]
    # combined words into single string per company
    for i in np.arange(len(company_words)):
        company_words[i] = ''.join([keyword + ' ' for keyword in company_words[i]])[:-1]
    # only keep unique
    company_names = np.unique(company_words)
    return company_names

def find_unemployment_cause(document):
    """
    """
    companies = list()
    #sentences = sent_tokenize(document)
    #words = [word_tokenize(element) for element in sentences]
    words = word_tokenize(document)
    # clear out anything with \ symbol
    words = [word for word in words if word.isalpha()]
    # get the correct tags for all the sentences
    #iob_tagged = [tree2conlltags(ne_chunk(pos_tag(element))) for element in words]
    iob_tagged = tree2conlltags(ne_chunk(pos_tag(words)))
    #tree = ne_chunk(pos_tag(words))
    # get noun trees
    word_types = np.array(iob_tagged)[:,1]
    # get first two letters of type
    word_noun = np.array(words)[np.where([(element == 'NN')|(element == 'NNS') for element in word_types])[0]]
    word_noun=  np.unique(word_noun)
    return word_noun

def find_percents(document):
    """
    """
    percents = list()
    # get words
    words = document.split()
    # retrieve all numbers that arent years
    # only look at first char bc could contain . or %
    numbers = [word for word in words if (word[0].isnumeric())&(len(word) != 4)]
    
    return numbers
    
#potentials = company_names2
def get_best_result_ceo(keywords_search, potentials):
    """
    Does a Solr search on given keywords + potential answers.  The potential
    answer with the highest score is the answer
    """
    max_scores = list()
    for i in potentials:
        # create the corrected search term
        edited_search  = keywords_search.split()
        edited_search[0] = '+' + edited_search[0].replace(':',':"') + ' ' + i + '"' + '~10'
        edited_search[2] = '+' + edited_search[2].replace(':',':"') + ' ' + i + '"' + '~10^3'

        edited_search = ''.join([keyword + ' ' for keyword in edited_search])
        #results = solr.search('+title:"CEO'+' '+i+'"~10  title:company +title:"Berkshire Hathaway"',fl='score')
        results = solr.search(edited_search,fl='score')
        max_scores.append(results['response']['maxScore'])
    # take the highest max_score
    output = potentials[np.argmax(max_scores)]
    return output

def check_type(output):
    """
    """
    # initialize final
    final_output = list()
    output = [element for element in output if element.lower() not in stopwords]
    for answer in output:
        result = solr.search('title:'+answer)
        document = result['response']['docs'][0]['title'][0]
        sentences = sent_tokenize(document)
        # find out correct sentence
        answer_sent_index = np.where([answer in sentence for sentence in sentences])[0]
        

        try:    
            words = word_tokenize(sentences[answer_sent_index[0]])
            ne_tree = ne_chunk(pos_tag(words))
            iob_tagged = tree2conlltags(ne_tree)
            # find what tag is in other context
            # get answer index
            try:
                answer_index = list(np.array(iob_tagged)[:,0]).index(answer.split()[0])
                answer_type = iob_tagged[answer_index][1:]
                # if the answer is a proper noun and organization, it is a good answer type
                if (answer_type[0] == 'NNP')&(answer_type[1].split('-')[1] == 'ORGANIZATION'):
                    final_output.append(answer)
            except ValueError:
                'nothing'
            
        except IndexError:
            'nothing'
    return final_output
        

        

def get_best_result_company(keywords_search, potentials):
    """
    """
    max_scores = list()
    for i in potentials:
        # create the corrected search term
        edited_search  = keywords_search.split()
        edited_search[1] = '+' + edited_search[1].replace(':',':"') + ' ' + i + '"' + '~20^3'
        # treat 3 and 4 differntly bc we only need one of these (each represent a date in diff forms)
        edited_search[3] = edited_search[3].replace(':',':"') + ' ' + i + '"' + '~20^3'
        #edited_search[4] = edited_search[4].replace(':',':"') + ' ' + i + '"' + '~20)'
        edited_search[5] = '+' + edited_search[5].replace(':',':"') + ' ' + i + '"' + '~20^3'

        edited_search = ''.join([keyword + ' ' for keyword in edited_search])
        
        #results = solr.search('+title:"CEO'+' '+i+'"~10  title:company +title:"Berkshire Hathaway"',fl='score')
        results = solr.search(edited_search,fl='score')
        max_scores.append(results['response']['maxScore'])
    # take the highest max_score
    #output = potentials[np.argmax(max_scores)]
    output = potentials[np.where(np.array(max_scores) >5)]
    print(np.array(max_scores)[np.where(np.array(max_scores) >5)])
    print(output)
    # filter output again to make sure just companies
    output = np.unique(check_type(output))
    print(output)
    return output

def clean_nouns_causes(output):
    """
    """
    porter_stemmer = PorterStemmer()
    stems = [porter_stemmer.stem(element) for element in output]
    # initialize final
    final_output = list()
    output = [element for element in output if element.lower() not in stopwords]
    for answer in output:
        result = solr.search('title:'+answer)
        document = result['response']['docs'][0]['title'][0]
        sentences = sent_tokenize(document)
        # find out correct sentence
        answer_sent_index = np.where([porter_stemmer.stem(answer) in [porter_stemmer.stem(element) for element in word_tokenize(sentence)] for sentence in sentences])[0]

        try:    
            words = word_tokenize(sentences[answer_sent_index[0]])
            ne_tree = ne_chunk(pos_tag(words))
            iob_tagged = tree2conlltags(ne_tree)
            # find what tag is in other context
            # get answer index
            try:
                stems = [porter_stemmer.stem(element) for element in np.array(iob_tagged)[:,0]]
                answer_index = list(stems).index(porter_stemmer.stem(answer))
                answer_type = iob_tagged[answer_index][1:]
                # if the answer is a proper noun and organization, it is a good answer type
                if (answer_type[0][:2] == 'NN'):
                    final_output.append(answer)
            except ValueError:
                'nothing'
            
        except IndexError:
            'nothing'
    return final_output

    
    

#potentials = nouns
def get_best_result_cause(keywords_search, potentials):
    """
    """
    max_scores = list()
    # make sure no stopwords
    potentials = [element for element in potentials if element.lower() not in stopwords]

    for i in potentials:
        edited_search = keywords_search.split()
        # add word within 20 words of GDP
        # edited_search[2] = edited_search[2].replace(':',':"') + ' ' + i + '"' + '~20'
        edited_search[0] = '+' + edited_search[0].replace(':',':"') + ' ' + i + '"' + '~5'
        edited_search[1] = edited_search[1].replace(':',':"') + ' ' + i + '"' + '~5'
        edited_search[2] = edited_search[2].replace(':',':"') + ' ' + i + '"' + '~5'
        edited_search[3] = edited_search[3][:-1].replace(':',':"') + ' ' + i + '"' + '~5'
        edited_search[3] = edited_search[3] + ')^3'
        edited_search.append('+' + edited_search[4].replace(':',':"') + ' ' + i + '"' + '~5^3')
        #edited_search.append(edited_search[4].replace(':',':"') + ' ' + i + '"' + '~10^5')
        #edited_search.append(edited_search[4].replace(':',':"') + ' ' + i + '"' + '~20^2')
        #edited_search.append(edited_search[2].replace(':',':"') + ' ' + i + '"' + '~50^2')
        edited_search[4] = '+' + edited_search[4] + '^10'
        # edited_search.append('(title:increase title:decrease)^2')
        #edited_search.append(' title:'+i + '^5')
        edited_search = ''.join([keyword + ' ' for keyword in edited_search])[:-1]
        results = solr.search(edited_search,fl='score')
        max_scores.append(results['response']['maxScore'])
    output = np.array(potentials)[np.argsort(max_scores)[-100:]]
    #np.array(potentials)[np.where(np.array(max_scores) >10)[0]]
    #print(output)
    #print(np.array(max_scores)[np.where(np.array(max_scores) >2)])
    # filter out the shit
    final_output = clean_nouns_causes(output)
    print(final_output)
    return final_output

def get_best_result_percent(keywords_search,potentials):
    """
    """
    max_scores = list()

    # get rid of things with disallowed characters
    potentials = [element for element in potentials if '"' not in element]
    potentials = [element for element in potentials if ')' not in element]
    potentials = [element for element in potentials if '(' not in element]
    potentials = [element for element in potentials if ':' not in element]
    potentials = [element for element in potentials if ']' not in element]
    potentials = [element for element in potentials if 'Q' not in element]
    potentials = [element for element in potentials if ',' not in element]
    potentials = [element for element in potentials if '[' not in element]
    for i in potentials:
        edited_search = keywords_search.split()
        # add word within 20 words of GDP
        # edited_search[2] = edited_search[2].replace(':',':"') + ' ' + i + '"' + '~20'
        edited_search[0] = '+' + edited_search[0].replace(':',':"') + ' ' + i + '"' + '~5'
        edited_search[1] = edited_search[1].replace(':',':"') + ' ' + i + '"' + '~5'
        edited_search[2] = edited_search[2].replace(':',':"') + ' ' + i + '"' + '~5'
        edited_search[3] = edited_search[3][:-1].replace(':',':"') + ' ' + i + '"' + '~5'
        edited_search[3] = edited_search[3] + ')'
        edited_search.append('+' + edited_search[4].replace(':',':"') + ' ' + i + '"' + '~10')
        #edited_search.append(edited_search[4].replace(':',':"') + ' ' + i + '"' + '~10^5')
        #edited_search.append(edited_search[4].replace(':',':"') + ' ' + i + '"' + '~20^2')
        #edited_search.append(edited_search[2].replace(':',':"') + ' ' + i + '"' + '~50^2')
        edited_search[4] = '+' + edited_search[4] 
        edited_search[5] = '+' + edited_search[5]+'^10'
        edited_search[6] = edited_search[6].replace(':',':"') + ' ' + i + '"' + '~5'
        edited_search[7] = edited_search[7].replace(':',':"') + ' ' + i + '"' + '~5'
        edited_search[8] = edited_search[8][:-1].replace(':',':"') + ' ' + i + '"' + '~5'
        edited_search[8] = edited_search[8] + ')'
        edited_search.append('title:' + i + '^5')
        # edited_search.append('(title:increase title:decrease)^2')
        #edited_search.append(' title:'+i + '^5')
        edited_search = ''.join([keyword + ' ' for keyword in edited_search])[:-1]
        results = solr.search(edited_search,fl='score')
        max_scores.append(results['response']['maxScore'])
    output = potentials[np.argmax(max_scores)]
    #np.array(potentials)[np.argsort(max_scores)[-10:]]
    return output
    
# q2 = 'Which companies went bankrupt in month November of year 2008?'
q3 = 'What affects GDP'
# 53.7
q3 = 'What percentage of drop or increase is associated with output?'

# 10.3
q3 = 'What percentage of drop or increase is associated with sanctions?'
# 1.67%
q3 = 'What percentage of drop or increase is associated with spending?'

 
#question = q3
def get_answer(question,keywords_search_last):
    """
    """
    q_words = word_tokenize(question)
    if q_words[0]== 'Who':
        ne_tree = ne_chunk(pos_tag(q_words))
        iob_tagged = tree2conlltags(ne_tree)
        # designate keywords of the question
        # nouns and anything in IOB bag are key
        # nouns = [element[0] for element in iob_tagged if element[1] == 'NN']
        # anything in IOB tag
    
        #initialize words used
        # maybe try making a single term of the BI tagged phrases too.  or try making
        # them closer together
        keywords = list()
        for i in np.arange(len(iob_tagged)):
            if ((iob_tagged[i][1][:2] == 'NN')|(iob_tagged[i][2][0] in set(['I','B'])))&( iob_tagged[i][0] not in stopwords):
                keywords.append(iob_tagged[i][0])
        # get top document candidates
        # transform keywords into search query
        keywords_search = ''.join(['title:' + keyword + ' ' for keyword in keywords])
        
        results = solr.search(keywords_search,fl='* score')
        #results = solr.search('title:CEO  title:company +title:"Berkshire Hathaway"',fl='* score')
        #results = solr.search('title:"Berkshire Hathaway"',fl='* score')
        people_names = [find_name_document(results['response']['docs'][i]['title'][0]) for i in np.arange(10)]
        people_names = np.concatenate(people_names)
        # retrieve the top result
        person = get_best_result_ceo(keywords_search,people_names)
        print('the ceo is', person)
        return person, keywords_search
    if q_words[0] == 'Which':
        # get list of month abbr
        months = list(['','January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])    
        ne_tree = ne_chunk(pos_tag(q_words))
        iob_tagged = tree2conlltags(ne_tree)
        # extract important words in question
        keywords = list()
        for i in np.arange(len(iob_tagged)):
            if ((iob_tagged[i][1][:2] in set(['NN','CD', 'RB']))|(iob_tagged[i][2][0] in set(['I','B'])))&( iob_tagged[i][0] not in stopwords)&(iob_tagged[i][0] != iob_tagged[0][0]):
                keywords.append(iob_tagged[i][0])
        """
        # if entry is month, add a number of the month right after and vice versa
        tried adding, but hurt results
        if keywords[3] in set(months):
            keywords.insert(3,months.index(keywords[3]))
            #keywords[3] = list([keywords[3],months.index(keywords[3])])
        elif keywords[3] in set(np.arange(1,13)):
            keywords.insert(3,months[keywords[3]])
        """
        #keywords[5] = keywords[5] + '^5'
        #keywords[3] = keywords[3] + '^5'
        keywords_search = ''.join(['title:' + str(keyword) + ' ' for keyword in keywords])
        
        
        results = solr.search(keywords_search,fl='* score')
        # get company names
        company_names = [find_company_document(results['response']['docs'][i]['title'][0]) for i in np.arange(10)]
        company_names = np.concatenate(company_names)
        bankrupt_companies = get_best_result_company(keywords_search, company_names)
        print('the bankrupt companies are', bankrupt_companies)
        return bankrupt_companies, keywords_search
    if q_words[0] == 'What':
        if q_words[1] == 'affects':
            ne_tree = ne_chunk(pos_tag(q_words))
            iob_tagged = tree2conlltags(ne_tree)
            # extract important words in question
            keywords = list()
            for i in np.arange(len(iob_tagged)):
                if ((iob_tagged[i][1][:2] in set(['NN','CD', 'RB', 'VB']))|(iob_tagged[i][2][0] in set(['I','B'])))&( iob_tagged[i][0] not in stopwords)&(iob_tagged[i][0] != iob_tagged[0][0]):
                    keywords.append(iob_tagged[i][0])
            # get synonyms for affect
            syns = wordnet.synsets(keywords[0])
            synonyms = [element.name() for element in syns[1].lemmas()]
            # only the first two look great, so use those
            keywords_synonyms = ''.join(['title:' + str(keyword) + ' ' for keyword in (synonyms[:2]+list(['increase','decrease']))])[:-1]
            # make full search term
            keywords_search = '(' +keywords_synonyms + ')' + ' title:' + keywords[1]
            results = solr.search(keywords_search,fl='* score',rows=30)
            # get potential causes
            nouns = [find_unemployment_cause(results['response']['docs'][i]['title'][0]) for i in np.arange(30)]
            nouns = np.concatenate(nouns)
            porter_stemmer = PorterStemmer()
            # stem
            nouns = [word.lower() for word in nouns]
            nouns = np.unique(nouns)
            causes = get_best_result_cause(keywords_search, nouns)
            print('The following affect GDP:', causes)
            return causes,keywords_search
            
            
        if q_words[1] == 'percentage':
            
            porter_stemmer = PorterStemmer()
            ne_tree = ne_chunk(pos_tag(q_words))
            iob_tagged = tree2conlltags(ne_tree)
            # extract important words in question
            keywords = list()
            for i in np.arange(len(iob_tagged)):
                if ((iob_tagged[i][1][:2] in set(['NN','CD', 'RB', 'VB']))|(iob_tagged[i][2][0] in set(['I','B'])))&( iob_tagged[i][0] not in stopwords)&(iob_tagged[i][0] != iob_tagged[0][0]):
                    keywords.append(iob_tagged[i][0])
            
            # group synonyms
            syn_one = '(' + 'title:' +keywords[0] + ' ' + 'title:' +'percent' + ' ' + 'title:' +'%' ')'
            """
            syns = wordnet.synsets(keywords[3])
            syn_two = '(' + 'title:' +keywords[1] + ' '+'title:' +'decrease' + ' '+'title:' +keywords[2] + ')'
            syn_three = 
            """
            # use the keywords from last search with the subject added and percent syns
            keywords_search = keywords_search_last + ' ' + 'title:' + keywords[-1] + ' ' + syn_one
            # do search
            results = solr.search(keywords_search,fl='* score',rows=50)
            # take the potential answers
            percents = [find_percents(results['response']['docs'][i]['title'][0]) for i in np.arange(50)]
            percents = np.concatenate(percents)
            # just unique
            percents = np.unique(percents)
            # clean any numbers that end in punctuation
            # find if last punc
            last_bad = np.where([element[-1] in list(['.','?','!','"']) for element in percents])[0]
            # if any, get rid of last char
            if len(last_bad) > 0:
                for index in last_bad:
                    percents[index] = percents[index][:-1]
            # find the best answer
            percent = get_best_result_percent(keywords_search,percents)
            print('The given factor affects GDP by the following percent:', percent)
            return percent, keywords_search
        
q1 = 'Who is the CEO of company Berkshire Hathaway'
print(q1)
ceo,keywords = get_answer(q1,0)
q1 = 'Who is the CEO of company Yahoo'
print(q1)
ceo,keywords = get_answer(q1,0)
q1 = 'Who is the CEO of company Amazon'
print(q1)
ceo,keywords = get_answer(q1,0)
# Enron not recognized
q2 = 'Which companies went bankrupt in month December of year 2001?'
print(q2)
companies,keywords = get_answer(q2,0)

#Blockbuster
# blockbuster recognized, but late
#q2 = 'Which companies went bankrupt in month November of year 2013?'

# GM
q2 = 'Which companies went bankrupt in month June of year 2009?'
print(q2)
companies,keywords = get_answer(q2,0)
# Lehman
q2 = 'Which companies went bankrupt in month September of year 2008?'
print(q2)
companies,keywords = get_answer(q2,0)
q3 = 'What affects GDP'
print(q3)
causes,keywords_gdp = get_answer(q3,0)

# 53.7
q3 = 'What percentage of drop or increase is associated with output?'
print(q3)
percent,keywords = get_answer(q3,keywords_gdp)
# 10.3
q3 = 'What percentage of drop or increase is associated with sanctions?'
print(q3)
percent,keywords = get_answer(q3,keywords_gdp)
# 1.67%
q3 = 'What percentage of drop or increase is associated with spending?'
print(q3)
percent,keywords = get_answer(q3,keywords_gdp)

       
    
# finish logging
sys.stdout = old_stdout

log_file.close()  
    

    


