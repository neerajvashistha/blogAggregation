from __future__ import absolute_import
from __future__ import print_function
import six
import sys
from modules.rake import *
import operator
import io,requests,re
# from modules.spellcheck import *
from spellchecker import SpellChecker

def sentence_correct(sentence):
    spell = SpellChecker()
    sentence = sentence.replace('/','-')
    misspelled = spell.unknown(sentence.split())
    for words in list(misspelled):
        correct = spell.correction(words)
        sentence = sentence.replace(words,correct)
    return sentence

def extract_entity(sentence):
    stoppath = "../data/corpus/SmartStoplist_mod.txt"
    R_object = Rake(stoppath)
    # 2. Split text into sentences  
    sentence = clean_text(sentence)
    #print(sentence)

    txt = sentence_correct(sentence)
    print("txt",txt)
    #print(txt)
    sentenceList = split_sentences(txt)
    # 3. generate candidate keywords
    stopwordpattern = build_stop_word_regex(stoppath)
    phraseList = generate_candidate_keywords(sentenceList, stopwordpattern)
    remainAttrib=set(sentence.lower().split())-set(' '.join(phraseList).split())
    remAttrib_final = Rem_Attrib_EXT("../data/corpus/SmartStoplist_mod.txt",remainAttrib)
    for i in range(len(remAttrib_final)):
        phraseList.append(remAttrib_final[i])
    if sort_return(sentence,phraseList).split():
        return sort_return(sentence,phraseList).split()
    else:
        return sentence.split()


def remove_tags(text):
    TAG_RE = re.compile(r'<[^>]+>')
    text = text.replace('\r','').replace('\n','')    
    return TAG_RE.sub('', re.sub('[^A-Za-z0-9.,:; ]+', '', text))

def Rem_Attrib_EXT(stoppath,Remlist):
    stoplist=list()
    Remlist_final=list()
    f2=open(stoppath,'r')
    for line in f2:
        w = line.split()
        for word in w:
            stoplist.append(word)
            #end 
    for word in Remlist:
        if word in stoplist: continue
        else: 
            Remlist_final.append(word)
    return Remlist_final
def clean_text(raw_text):
    raw_text = raw_text.upper()
    cleanText = ""
    if bool(re.search(r'([0-9]+ [GB])\w+',raw_text)):
        cleanText = re.sub(r'([0-9]+ [GB])\w+', ''.join(re.findall(r'([0-9]+ [GB])\w+',raw_text)[0].split())+'B',raw_text,flags=re.IGNORECASE)
    else:
        cleanText = raw_text
    return cleanText.lower()

def fetch_data(searchTerm):
    print('*************************************************')
    print(searchTerm)
    print('*************************************************')
    URL = "http://scandid-psolr.cloudapp.net:8983/solr/products/select?q=" + ' '.join(extract_entity(searchTerm)) + "&wt=json&indent=true&rows=5&start=0"
    response = requests.get(URL,timeout=300)
    print(response.text)

def sort_return(originalString, alist):
    astring = ' '.join(alist)
    tokenastring = astring.split(' ')
    tokenoriginal = originalString.lower().split(' ')
    s=""
    for i in tokenoriginal:
        for j in tokenastring:
            if i == j:
                s=s+j+" "
    return s.strip()

if __name__ == "__main__":
    print(extract_entity("mi 4"))
    print(extract_entity("moto x play"))
    print(extract_entity("collar tshirts"))
    print(extract_entity("Infinix Note 4"))
