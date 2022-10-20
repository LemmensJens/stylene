from math import log, sqrt, floor
from collections import Counter
from statistics import mean
from string import punctuation
import operator
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd

#BASELINE_SYLLABIFIER________________________________________________________________
def get_n_syllables(word):
    vowels = ['a', 'e', 'i', 'o', 'u', 'y']
    cnt=0
    for i, char in enumerate(word):
        previous_char = '-' if i==0 else word[i-1]
        if char in vowels and previous_char not in vowels:
            cnt+=1
    return cnt

#LEXICAL_RICHNESS_SCORES_____________________________________________________________
def ttr(n_types,n_tokens):
	return round(n_types/n_tokens, 5)

def rttr(n_types,n_tokens):
	return round(n_types/sqrt(n_tokens), 5)

def cttr(n_types,n_tokens):
	return round(n_types/sqrt(2*n_tokens), 5) 

def Herdan(n_types,n_tokens):
	return round(log(n_types)/log(n_tokens), 5)

def Summer(n_types,n_tokens):
	return round(log(log(n_types))/log(log(n_tokens)), 5)

def Dugast(n_types,n_tokens):
	return round((log(n_tokens)**2)/(log(n_tokens)-log(n_types)), 5)

def Maas(n_types,n_tokens):
	return round((log(n_tokens)-log(n_types))/(log(n_tokens)**2), 5) 

#READABILITY SCORES__________________________________________________________________
def ARI(n_char, n_tokens, n_sentences):
	return round(4.71*(n_char/n_tokens)+0.5*(n_tokens/n_sentences)-21.43, 5)

def ColemanLiau(tokens, tokenized_sentences):
	if len(tokens) < 100:
		return 0
	chunks = [tokens[i:i+100] for i in range(0, len(tokens), 100) if i+100<=len(tokens)]
	L = mean([len(''.join(chunk)) for chunk in chunks]) # avg. n char per 100 tokens
	S = len(tokenized_sentences)/len(tokens)*100 # avg. n sent per 100 tokens
	return round(0.0588*L-0.296*S-15.8, 5)

def Flesch(ASL, ASW):
	return round(206.835-(1.015*ASL)-(84.6*ASW), 5)

def Fog(ASL, syllables):
	syllables = [s for sent in syllables for s in sent]
	if len(syllables) < 100:
		return 0
	PHW = len([s for s in syllables if s >= 3])/len(syllables) # percentage hard words
	return round(0.4*(ASL + PHW), 5)

def Kincaid(ASL, ASW):
	return round((0.39*ASL)+(11.8*ASW)-15.59, 5)

def LIX(n_tokens, n_sentences, n_long_tokens):
	return round((n_tokens/n_sentences)+(n_long_tokens*100/n_tokens), 5)

def RIX(n_long_tokens, n_sentences):
	return round(n_long_tokens/n_sentences, 5)

def SMOG(sample):
	length = len(sample)
	if length < 30:
		return 0
	else:
		i = floor(length/3)
		sample = sample[:10] + sample[i:i+10] + sample[-10:]
		sample = [s for sent in sample for s in sent]
	n_polysyllabic = len([s for s in sample if s > 2])
	return round(3+sqrt(n_polysyllabic), 5)

#DISTRIBUTIONS_______________________________________________________________________
def get_word_length_distribution(tokens):
	"""returns {token_length: rel_freq}"""
	lengths = [len(t) for t in tokens]
	dist = dict(Counter(lengths))
	dist = {k:round(v/len(tokens), 5) for k,v in dist.items()}
	dist = {str(k)+'_character_tokens':v for k,v in dist.items()}
	dist = dict(sorted(dist.items(), key=operator.itemgetter(1),reverse=True))
	return dist

def get_grapheme_distribution(tokens):
	"""returns {grapheme: rel_freq}"""
	graphemes = ''.join(tokens)
	n_total = len(graphemes)
	dist = dict(Counter(graphemes))
	for k,v in dist.items():
		dist[k] = round(v/n_total, 5)
	dist = dict(sorted(dist.items(), key=operator.itemgetter(1),reverse=True))
	return dist

def get_word_internal_grapheme_profile(tokens):
	"""returns {grapheme: % of words that contain grapheme}"""
	graphemes = set(''.join(tokens))
	n_tokens = len(tokens)
	profile = {}

	for g in graphemes:
		for t in tokens:
			if g in t:
				if g in profile.keys():
					profile[g+'_word_internal'] += 1
				else:
					profile[g+'_word_internal'] = 1
	
	profile = {k:round(v/n_tokens, 5) for k,v in profile.items()}
	profile = dict(sorted(profile.items(), key=operator.itemgetter(1),reverse=True))
	return profile

def get_function_word_distribution(doc):
	"""returns {function_word: rel_freq}"""
	allowed_pos = {'ADP', 'AUX', 'CCONJ', 'DET', 'PART', 'PRON', 'SCONJ'}
	function_words = [w.text.lower() for s in doc.sentences for w in s.words if w.upos in allowed_pos]
	n_function_words = len(function_words)
	dist = dict(Counter(function_words))
	dist = {k:round(v/n_function_words, 5) for k,v in dist.items()}
	dist = {'function_word_'+k:v for k,v in dist.items()}
	dist = dict(sorted(dist.items(), key=operator.itemgetter(1),reverse=True))
	return dist

def get_grapheme_positional_freq(tokens):
	"""returns {char_idx_in_token: {char: rel_freq}"""
	n_tokens = len(tokens)
	lengths = [len(t) for t in tokens]
	longest = sorted(lengths)[-1]
	profile={}
	
	for i in range(1, longest+1):
		grapheme_freq = {}
		for t in tokens:
			if len(t) >= i:
				if t[i-1] in grapheme_freq.keys():
					grapheme_freq[t[i-1]] += 1
				else:
					grapheme_freq[t[i-1]] = 1

		grapheme_freq = {k:round(v/n_tokens, 5) for k,v in grapheme_freq.items()}
		grapheme_freq = {f'char_idx_{i}_{k}':v for k,v in grapheme_freq.items()}
		grapheme_freq = dict(sorted(grapheme_freq.items(), key=operator.itemgetter(1),reverse=True))
		profile['char_idx_'+str(i)] = grapheme_freq

	return profile

def get_punct_dist(text):
	"""returns {punctuation_mark: freq/total_num_characters, ...}"""
	dist = {}
	for p in punctuation:
		dist[p] = text.count(p)	

	n_char = len(text.replace(' ', ''))

	dist_by_char = {k:round(v/n_char, 5) for k,v in dist.items()}
	dist_by_char = dict(sorted(dist_by_char.items(), key=operator.itemgetter(1),reverse=True))
	return dist_by_char

def get_positional_word_profile(doc):
	"""returns {token_idx_in_sent: {word: rel_freq}}"""
	tokens = [[w.text.lower() for w in s.words] for s in doc.sentences]
	n_positions = max([len(s) for s in tokens])
	profile = {}

	for i in range(n_positions):
		k = i
		words = [s[k] for s in tokens if len(s)>k]
		n_sentences = len(words)
		v = dict(Counter(words))
		v = {k:v/n_sentences for k,v in v.items()}
		v = {f'token_idx_{str(i)}_{k}':v for k,v in v.items()}
		v = dict(sorted(v.items(), key=operator.itemgetter(1),reverse=True))
		profile['token_idx_'+str(k)] = v

	return profile

def get_ngram_profile(tokens, ngram_range):
	vec = TfidfVectorizer(use_idf=False, norm="l1")
	X = vec.fit_transform([' '.join(tokens)])
	profile = dict()
	for v,k in zip(X.toarray().flatten(), vec.get_feature_names()):
		profile[k] = v
	return profile
