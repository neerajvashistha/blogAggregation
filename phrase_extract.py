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

# def spellcorector(sentence):
#     spell = SpellChecker()
#     for tokens in list(spell.unknown(spell.split_words(sentence))):
#         correct = spell.correction(tokens)
#         sentence = sentence.replace(tokens,correct)
#     return sentence

def custom_ner(sentence):
    """for the purpose of phrase extraction this function is employed

    :param name: sentence
    :type name: str. 
    :param state: free from slangs and spell errors
    :type state: str 
    :returns: list -- extracted phrases. 
    :raises: AttributeError, KeyError

    """ 
    # 1. initialize RAKE by providing a path to a stopwords file
    stoppath = "modules/SmartStoplist_mod.txt"
    rake_object = rake.Rake(stoppath)
    # 2. Split text into sentences
    sentenceList = rake.split_sentences(sentence)
    # 3. generate candidate keywords
    stopwordpattern = rake.build_stop_word_regex(stoppath)
    phraseList = rake.generate_candidate_keywords(sentenceList, stopwordpattern)
    phraseList = [t for t in phraseList if '.' not in t and (len(t) > 6 or len(t.split()) > 2)]
    return phraseList

if __name__ == "__main__":
    print(extract_phrase("Samsung Phone"))