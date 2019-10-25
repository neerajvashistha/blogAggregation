#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re,itertools
from nltk.corpus import stopwords
flatten = itertools.chain.from_iterable
import spacy  
import numpy as np
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models import Phrases

from gensim.models import Phrases
from gensim.models.phrases import Phraser


# In[2]:


import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


# In[57]:


from phrase_extract import custom_ner
from random import shuffle


# In[4]:


stopword_list = stopwords.words('english')
stopword_list.extend(['google','facebook','twitter','linkedin','whatsapp'])


# In[5]:


df = pd.read_csv('raw_blog_content_cleaned.csv')
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
df.head()


# In[6]:


def preProcess_stage1(text):
    #print "original:", text
    # sentence segmentation - assume already done
    # Remove Emails
    text = re.sub(r'\S*@\S*\s?', '', text)
    # Remove website links
    text = re.sub(r'http[s]?://\S+', '', text)
    # Remove distracting single quotes
    text = re.sub(r"\'", "", text)
    # Remove distracting double quotes
    text = re.sub(r'\"', "", text)
    # Remove new line characters
    text = re.sub(r'\s+', ' ', text)
    # word normalisation
    text = re.sub(r"(\w)([.,;:!?'/\"”\)])", r"\1 \2", text)
    text = re.sub(r"([.,;:!?'/\"“\(])(\w)", r"\1 \2", text)
    # normalisation
    text = re.sub(r"(\S)\1\1+",r"\1\1\1", text)
    #tokenising
    
    tokens = list(flatten([re.split(r"\s+",t) for t in re.split('(\d+)',text)]))
    tokens = [re.sub(r'[^A-Za-z]+','',t) for t in tokens]
    tokens = [t.lower() for t in tokens]
    tokens = [t for t in tokens if t not in ' ']
    return tokens


# In[7]:


# corpus of all body data for NER
with open('corpus.txt','w') as f:
    allblog = df.body.values.tolist()
    for blog in allblog:
        sentList = []
        sentences = blog.split('.')
        for sent in sentences:
            tokens = preProcess_stage1(sent)
            sentList.append(' '.join(tokens))
        f.writelines('. '.join(sentList))

# stopword file for NER
stopword_file=open('stopword.txt','w')
stopword_file.writelines(stopword_list)
stopword_file.close()


# In[8]:


df['tokens_1'] = df.apply(lambda x : preProcess_stage1(str(x.body)),axis = 1)


# In[9]:


df['new_body'] = df.apply(lambda x : '. '.join([' '.join(preProcess_stage1(sent)) for sent in x.body.split('.')]),axis = 1)


# In[10]:


df['ner_terms'] = df.apply(lambda x : custom_ner(x.new_body),axis = 1)


# In[11]:


df['ner_terms']


# In[12]:


nlp = spacy.load('en', disable=['parser', 'ner'])

def remove_stopwords(tokens):
    return [t for t in tokens if t not in stopword_list]

def make_bigrams(tokens):
    tokens = [tokens]
    bigram = Phrases(tokens, min_count=1, threshold=2, delimiter=b' ')
    bigram_phraser = Phraser(bigram)
    bigram_tokens = list(flatten([bigram_phraser[sent] for sent in tokens]))
    bigrams_new = [t for t in bigram_tokens if len(t.split()) > 1 ]
#     bigrams_new = ['_'.join(t.split())  if len(t.split()) > 1 else t for t in bigram_tokens]
    return bigrams_new

def make_trigrams(tokens):
    tokens = [tokens]
    bigram = gensim.models.Phrases(tokens, min_count=1, threshold=2) 
    trigram = gensim.models.Phrases(bigram[tokens], threshold=2) 
    trigram_phraser = Phraser(trigram)
    trigram_tokens = list(flatten([trigram_phraser[sent] for sent in tokens]))
    trigrams_new = [t for t in trigram_tokens if len(t.split()) > 2 ]
    return trigrams_new

def lemmatization(tokens, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    doc = nlp(" ".join(tokens)) 
    return [token.lemma_ for token in doc if token.pos_ in allowed_postags]

def preProcess_stage2(tokens):
    # Stop Words
    clean_tokens = remove_stopwords(tokens)

    # BIgrams
    bigrams_tokens = make_bigrams(clean_tokens)

    # TRIgarms
    trigrams_tokens = make_trigrams(clean_tokens)
    
    # remove lemma with noun, adj, vb, adv words only
#     token_lemma = lemmatization(bigrams_tokens, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    token_lemma = lemmatization(clean_tokens, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    token_lemma = [t for t in token_lemma if len(t) > 2 ]
    return token_lemma + bigrams_tokens + trigrams_tokens


# In[13]:


df['tokens_2'] = df.apply(lambda x : preProcess_stage2(x.tokens_1),axis = 1)


# In[14]:


print(df['tokens_2'].head())


# In[15]:


token_2List = df.tokens_2.values.tolist()
token_2List = df.ner_terms.values.tolist()

# Create Dictionary
id2word = corpora.Dictionary(token_2List)

# Create Corpus
texts = token_2List

# Term Document Frequency BOW model
corpus = [id2word.doc2bow(text) for text in texts]


# In[18]:


# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
											id2word=id2word,
											num_topics=20, 
											random_state=100,
											update_every=1,
											chunksize=100,
											passes=10,
											alpha='auto',
											per_word_topics=True)


# In[19]:


# Build LDA Multicore model
lda_multicore = gensim.models.ldamulticore.LdaMulticore(corpus=corpus,
											num_topics=20,
											id2word=id2word,
											workers=3)


# In[20]:


# Build LDA Mallet model
mallet_path = 'mallet-2.0.8/bin/mallet' 
lda_mallet = gensim.models.wrappers.LdaMallet(mallet_path, 
											corpus=corpus, 
											num_topics=20, 
											id2word=id2word)

# In[22]:


# Compute Perplexity of LDA, lower the model perplexity the better it is
print('\nPerplexity of LDA: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score of LDA, higher the model coherence the better it is 
coherence_model_lda = CoherenceModel(model=lda_model, texts=token_2List, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score of LDA: ', coherence_lda)

# Compute Perplexity of LDA Multicore
print('\nPerplexity of LDA Multicore: ', lda_multicore.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score of LDA Multicore
coherence_model_lda = CoherenceModel(model=lda_multicore, texts=token_2List, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score of LDA Multicore: ', coherence_lda)


# Compute Coherence Score of LDA Mallet
coherence_model_lda = CoherenceModel(model=lda_mallet, texts=token_2List, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score of LDA Mallet: ', coherence_lda)

# the coherence score for lda_mallet is around .43 best out of all three models

optimal_model = lda_mallet
data = df.body.values.tolist()
def return_topic_per_blog(model, corpus, texts):
    blog_topics_df = pd.DataFrame()
    # Get main topic in each document
    for i, row in enumerate(model[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True) # sortin to get the best topic first
        # Get keywords and percentage contribution for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = model.show_topic(topic_num)
                keywords = ", ".join([word for word, prop in wp])
                blog_topics_df = blog_topics_df.append(pd.Series([keywords, round(prop_topic,4)]), ignore_index=True)
            else:
                break
    blog_topics_df.columns = ['Topic_Keywords', 'Perc_Contribution']

    # Add blog to the dataframe
    blogs = pd.Series(texts)
    blog_topics_df = pd.concat([blog_topics_df, blogs], axis=1)
    return(blog_topics_df)


df_keywords = return_topic_per_blog(optimal_model, corpus, data)

# Format
df_topic = df_keywords.reset_index()
df_topic.columns = ['Document_No', 'Topic_Keywords', 'Perc_Contrib', 'body']

# Show
df_topic.head(50)


# In[26]:


# combine enitre data
com = pd.merge(df, df_topic, on='body')


# In[27]:

# write it to csv 
com.to_csv('lda_keyword.csv',index=False)


# In[28]:
print("Preparing and loading Google word2vec....")
# map the generated topics to actual topics using similarity metrics
from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)


# In[29]:


topics_scores = {
    "marketing" : 0.0,
    "branding" : 0.0,
    "growth marketing" : 0.0,
    "brand development" : 0.0,
    "growth strategies" : 0.0,
    "product management" : 0.0,
    "product discovery" : 0.0,
    "product growth" : 0.0,
    "product management fundamentals" : 0.0,
    "agile principles" : 0.0,
    "company culture" : 0.0,
    "company growth" : 0.0,
    "people management" : 0.0,
    "startup fundamentals" : 0.0,
    "interpersonal skills" : 0.0,
    "business fundamentals" : 0.0,
    "business growth" : 0.0,
    "sales growth" : 0.0,
    "investment cycle" : 0.0
}


# In[61]:


def distance_metrics(x):
    #tokens = topics.split(',')
    topics = x.ner_terms
    for k,v in topics_scores.items():
        distance = []
        for t in topics:           
            #calculate distance between two sentences using WMD algorithm
            distance.append(model.wmdistance(k, t))
        topics_scores[k] = np.mean(distance)
    sorted_weight = sorted(topics_scores.items(), key=lambda x:x[1], reverse=True)
    return sorted_weight

def most_rated_topics(x):
    title_ner = custom_ner(x.title)
    final_tokens = []
    if title_ner:
        for t in title_ner:
            final_tokens.append(' '.join(preProcess_stage1(t)))
    #print(final_tokens)
    top_rated = list(x.assorted_topic_scores)[:5] 
    top_rated = final_tokens + [i[0] for i in top_rated]
    shuffle(top_rated)
    #print(top_rated)
    return ', '.join(top_rated)


# In[33]:

print("Computing phrase similarity between predicted phrases and required topics")
com['assorted_topic_scores'] = com.apply(lambda x : distance_metrics(x), axis = 1 )



# In[62]:

print("Sorting and rearranging topics based on distnace metrics ")
com['topic'] = com.apply(lambda x : most_rated_topics(x), axis = 1)



# In[37]:

# write the imporved version to csv
com.to_csv('lda_keyword.csv',index=False)


# In[65]:
print("Writing to JSON articles_topic.json")
# return back JSON 
com[['title','url','body','topic']].to_json('articles_topic.json',orient='records')




