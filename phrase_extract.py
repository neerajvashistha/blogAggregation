from __future__ import absolute_import
from __future__ import print_function
import six
import sys
from modules import rake
import operator
import io
# from modules import spellcheck
# from spellchecker import SpellChecker
from nltk.corpus import stopwords
stopword_list = stopwords.words('english')
stopword_list.extend(['google','facebook','twitter','linkedin','whatsapp'])

def spellcorector(sentence):
    spell = SpellChecker()
    for tokens in list(spell.unknown(spell.split_words(sentence))):
        correct = spell.correction(tokens)
        sentence = sentence.replace(tokens,correct)
    return sentence

def extract_phrase(sentence):
    """for the purpose of phrase extraction this function is employed

    :param name: sentence
    :type name: str. 
    :param state: free from slangs and spell errors
    :type state: str 
    :returns: list -- extracted phrases. 
    :raises: AttributeError, KeyError

    """ 
    # 1. initialize RAKE by providing a path to a stopwords file
    stoppath = "/home/nv/blogAggregation/modules/SmartStoplist_mod.txt"
    rake_object = rake.Rake(stoppath)
    text = "I would like to order 2 mnchurien and rice. Send me a mechnic"
    # 2. Split text into sentences
    # txt = spellcheck.sentence_correct(sentence)
    # txt = spellcorector(sentence)
    sentenceList = rake.split_sentences(text)
    # 3. generate candidate keywords
    stopwordpattern = rake.build_stop_word_regex(stoppath)
    phraseList = rake.generate_candidate_keywords(sentenceList, stopwordpattern)
    return phraseList

if __name__ == "__main__":
    print(extract_phrase("Samsung Phone"))
    print(extract_phrase("Galaxy Phone"))
    print(extract_phrase("Samsung Galaxy"))
    print(extract_phrase("Samsung"))
    print(extract_phrase("Galaxy Samsung"))
    print(extract_phrase("Search for Galaxy S6 in Samsung phones with cases"))
    print(extract_phrase("Search for Galaxy S6 black in Samsung phones"))
    print(extract_phrase("Search for Samsung Galaxy S6 black with head phones"))
    print(extract_phrase("Search for Samsung Galaxy S6 black with headphones"))
    print(extract_phrase("Search for Samsung Galaxy S6 black with earphones"))
    print(extract_phrase("vibe k5 note"))
    
