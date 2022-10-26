import os, sys
from configparser import ConfigParser
from statistics import mean, stdev
from collections import Counter
from tqdm import tqdm

import pandas as pd
import numpy as np
import spacy
from stylene import util
from stylene.dictfeaturizer import DictFeaturizer

from string import punctuation

import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
import regex, spacy, emoji
import joblib, pickle
import json
import random
import matplotlib.pyplot as plt
import emoji

from statistics import mean, stdev
from pattern.nl import sentiment
from tqdm import tqdm

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.base import TransformerMixin,BaseEstimator
from sklearn.calibration import CalibratedClassifierCV

#______________________________________________________________________________________________

def stylene(inpt):
    """Contains the styelene pipeline. Input is a string provided by the user."""
#LOAD DATA_____________________________________________________________________________________
    text = ' '.join(inpt.split())
    feature_df = pd.DataFrame()
    liwc_df = pd.DataFrame()
    stats_df = pd.DataFrame()

#PREPROCESSING AND FEATURE EXTRACTION_____________________________________________________________________
    nlp = spacy.load('nl_core_news_sm')  
    doc = nlp(text)

    parsed_sentences = [[(w.text, w.pos_) for w in s] for s in doc.sents]
    pos_tags = [w.pos_ for w in doc]
    tokenized_sentences = [[w.text for w in s if w.pos not in {'PUNCT', 'SYM', 'X'}] for s in doc.sents]
    tokens = [w for w in doc if w.pos_ not in {'PUNCT', 'SYM', 'X'}]
    types = set([str(t).lower() for t in tokens])
    syllables = [[util.get_n_syllables(t) for t, pos in s if pos not in {'PUNCT', 'SYM', 'X'}] for s in parsed_sentences]
    punct = ' '.join(c for c in text if c in punctuation)

#STATISTICS____________________________________________________________________________________
    #LENGTH STATISTICS_____________________________________________________________________________
    n_char = len(text)
    n_syllables = sum([syl for sent in syllables for syl in sent])
    n_polysyllabic = len([i for sent in syllables for i in sent if i > 1])
    n_tokens = len(tokens)
    n_longer_than_6_char = len([t for t in tokens if len(t) > 6])
    n_types = len(types)
    n_sentences = len([s for s in doc.sents])

    avg_char_per_word = round(mean([len(t) for t in tokens]), 5)
    if len(tokens) > 1:
        std_char_per_word = round(stdev([len(t) for t in tokens]), 5)
    else:
        std_char_per_word = 0

    word_length_distribution = util.get_word_length_distribution(tokens)

    avg_syl_per_word = round(mean([s for sent in syllables for s in sent]), 5)
    if len(syllables) > 1:
        std_syl_per_word = round(stdev([s for sent in syllables for s in sent]), 5)
    else:
        std_syl_per_word = 0

    avg_t_per_s = round(mean([len(s) for s in tokenized_sentences]), 5)
    if n_sentences > 1:
        std_words_per_sent = round(stdev([len(s) for s in tokenized_sentences]), 5)
    else:
        std_words_per_sent = 0
    ratio_long_words = round(n_longer_than_6_char/n_tokens, 5)

    stats = {
    'Gemiddeld aantal karakters per woord': avg_char_per_word,
    'Gemiddeld aantal lettergrepen per woord': avg_syl_per_word,
    'Gemiddeld aantal woorden per zin': avg_t_per_s,
    'Verhouding lange woorden (7 of meer karakters)': ratio_long_words,
    }

    #LEXICAL RICHNESS______________________________________________________________________________
    ttr = util.ttr(n_types, n_tokens)
    lr = {'Type-token ratio': round(ttr, 2)}

    #READABILITY___________________________________________________________________________________
    ARI = util.ARI(n_char, n_tokens, n_sentences)
    readability = {'Automated readability index (ARI)': round(ARI, 1)}

    #DISTRIBUTIONS_________________________________________________________________________________
    punct_dist = util.get_punct_dist(text)
    pos_profile = util.get_ngram_profile(pos_tags, ngram_range=(1,1))
    
    dist = {
        'punctuation_distribution': punct_dist,
        'pos_profile': pos_profile,
    }

    #LIWC__________________________________________________________________________________________
    liwc = DictFeaturizer.load("./stylene/LIWC_Dutch.csv", relative=False)
    liwc_features = liwc.transform([str(t).lower() for t in tokens])

    def cname_to_dutch(k):
        if k=='sports':
            return 'liwc_sport'
        elif k=='music':
            return 'liwc_muziek'
        elif k=='relig':
            return 'liwc_geloof'
        elif k=='leisure':
            return 'liwc_vrije tijd'
        elif k=='job':
            return 'liwc_job'
        elif k=='school':
            return 'liwc_school'
        elif k=='family':
            return 'liwc_familie'
        else:
            return None

    #ADD COMPUTATIONS TO FEATURE DF______________________________________________________________
    feature_row = dict()
    stats_row = dict()
    liwc_row = dict()

    for k, v in stats.items():
        feature_row[k] = v
        stats_row[k] = v
    for k, v in lr.items():
        feature_row[k] = v
        stats_row[k] = v
    for k, v in readability.items():
        feature_row[k] = v
        stats_row[k] = v
    for k, v in liwc_features.items():
        feature_row['liwc'+k] = v
        if k in {'sports', 'music', 'relig', 'leisure', 'job', 'school', 'family'}:
            liwc_row[cname_to_dutch(k)] = v

    feature_df = feature_df.append(feature_row, ignore_index=True)
    liwc_df = liwc_df.append(liwc_row, ignore_index=True)
    stats_df = stats_df.append(stats_row, ignore_index=True)

    pattern_output = sentiment(text)
    
    columns_to_cast_to_integers = [c for c in feature_df.columns if c[:2]=='n_']
    stats_df = stats_df.astype({k: 'int' for k in columns_to_cast_to_integers})
    feature_df = feature_df.astype({k: 'int' for k in columns_to_cast_to_integers})

    #ADDITIONAL FEATURES FOR AUTHORSHIP ATTRIBUTION MODELS____________________________________
    feature_df['text'] = ' '.join(t.text for t in doc)
    feature_df['tokenized_text'] = ' '.join(t.text for t in doc)
    feature_df['pos'] = ' '.join(pos_tags)
    feature_df['punct'] = punct
    feature_df['ttr'], feature_df['ARI'], feature_df['avg_t_per_s'] = ttr, ARI, avg_t_per_s 
    feature_df['sentiment'], feature_df['subjectivity'] = pattern_output[0], pattern_output[1]
    feature_df['emoji'] = " ".join([char for char in text if char in emoji.EMOJI_DATA])
    feature_df['n_hashtags'] = len([t for t in text.split() if t[0]=="#"])
    feature_df['n_hyperlinks'] = feature_df['text'].str.count(r'(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w\.-]*)*\/?\S')

    #GENERATE GRAPHS___________________________________________________________________________
    def LiwcSpider(df):
        """Accepts df with LIWC features as input, returns spider graph"""
        theta = [c.lstrip('liwc_') for c in df.columns if 'liwc_' in c]
        new_df = df.rename(columns=lambda x: x.lstrip('liwc_'))

        fig = px.line_polar(new_df, r=df.iloc[0].to_list(), theta=theta, line_close=True, title='Onderwerp', color_discrete_sequence=['rgb(0,46,101)'])
        fig.update_traces(fill='toself')
        fig.update_layout(title_text='Figuur 2: Aantal gevonden woorden per onderwerp.', title_x=0.5, title_font=dict(size=12), title_font_family='Trebuchet MS', font_family='Trebuchet MS', font=dict(color='#002e65'))
        return fig

    def PosTable(df):

        transposed_df = df.transpose()
        transposed_df = transposed_df.rename(columns={0: 'Relatieve hoeveelheid'})
        transposed_df['Woordsoort'] = transposed_df.index
        transposed_df = transposed_df[transposed_df['Relatieve hoeveelheid']!=0]
        transposed_df = transposed_df.sort_values(by=['Relatieve hoeveelheid'], ascending=False)

        #generate table
        cell_height = 24
        n_rows = len(transposed_df)

        fig = go.Figure(
            data=[go.Table(
                header=dict(
                    values=['Woordsoort', 'Relatieve hoeveelheid'],
                    fill_color='#002e65', 
                    font=dict(color='white'),
                    font_family='Trebuchet MS',
                    align='left'),
                cells=dict(
                    values=[transposed_df['Woordsoort'], transposed_df['Relatieve hoeveelheid']],
                    align='left',
                    height=24,
                    font_family='Trebuchet MS',
                    font=dict(color='#002e65',)))
        ])

        fig.update_layout(
            title_text='Figuur 6: Relatieve hoeveelheid per woordsoort.', 
            title_x=0.5, 
            title_font=dict(size=12), 
            title_font_family='Trebuchet MS',
            margin=dict(l=0, r=0, t=50, b=0),
            height=cell_height*n_rows+100
            )

        return fig

    def PunctTable(df):

        transposed_df = df.transpose()
        transposed_df = transposed_df.rename(columns={0: 'Relatieve hoeveelheid'})
        transposed_df['Leesteken'] = transposed_df.index
        transposed_df = transposed_df[transposed_df['Relatieve hoeveelheid']!=0]
        transposed_df = transposed_df[transposed_df['Relatieve hoeveelheid']!=0]
        transposed_df = transposed_df.sort_values(by=['Relatieve hoeveelheid'], ascending=False)

        cell_height=24
        n_rows=len(transposed_df)

        if len(transposed_df) == 0:

            fig = go.Figure(
                data=[go.Table(
                    cells=dict(
                        values=["<i>Geen leestekens gevonden.</i>"], 
                        height=cell_height,
                        font_family='Trebuchet MS',
                        font=dict(color='#002e65'),
                        align="center"
            ))])

            fig.update_layout(
                title_text='Figuur 7: Relatieve hoeveelheid per leesteken (in verhouding tot het totaal aantal karakters).', 
                title_x=0.5,
                title_font=dict(size=12), 
                title_font_family='Trebuchet MS',
                margin=dict(l=0, r=0, t=50, b=0),
                height=cell_height*n_rows+100
            )

            return fig

        #generate table
        cell_height = 24
        n_rows = len(transposed_df)

        fig = go.Figure(
            data=[go.Table(
                header=dict(
                    values=['Leesteken', 'Relatieve hoeveelheid'],
                    fill_color='#002e65', 
                    font=dict(color='white'),
                    font_family='Trebuchet MS',
                    align='left'),
                cells=dict(
                    values=[transposed_df['Leesteken'], transposed_df['Relatieve hoeveelheid']],
                    align='left',
                    height=24,
                    font_family='Trebuchet MS',
                    font=dict(color='#002e65',)))
        ])

        fig.update_layout(
            title_text='Figuur 7: Relatieve hoeveelheid per leesteken (in verhouding tot het totaal aantal karakters).',  
            title_x=0.5, 
            title_font=dict(size=12), 
            title_font_family='Trebuchet MS',
            margin=dict(l=0, r=0, t=50, b=0),
            height=cell_height*n_rows+100
            )

        return fig
    
    def StatisticsTable(df):

        # drop LIWC columns, transpose, rename columns
        transposed_df = df.transpose()
        transposed_df = transposed_df.rename(columns={0: 'Score'})
        transposed_df['Statistiek'] = transposed_df.index

        #generate table
        cell_height = 24
        n_rows = len(transposed_df)
        
        fig = go.Figure(
            data=[go.Table(
                header=dict(
                    values=list(transposed_df.columns),
                    fill_color='#002e65', 
                    font=dict(color='white'),
                    font_family='Trebuchet MS',
                    align='left'),
                cells=dict(
                    values=[transposed_df.Statistiek, transposed_df.Score.apply(lambda x: round(x, 2))],
                    align='left',
                    height=cell_height,
                    font_family='Trebuchet MS',
                    font=dict(color='#002e65',)))
        ])

        fig.update_layout(
            title_text='Figuur 5: Algemene statistieken.', 
            title_x=0.5,
            title_font=dict(size=12), 
            title_font_family='Trebuchet MS',
            margin=dict(l=0, r=0, t=50, b=0),
            height=cell_height*n_rows+100
            )

        return fig

    pos_df = pd.DataFrame(data={k:[round(v, 3)] for k, v in pos_profile.items()})
    pos_table = PosTable(pos_df)

    punct_df = pd.DataFrame(data={k:[round(v, 5)] for k,v in punct_dist.items()})
    punct_table = PunctTable(punct_df)

    statistics_table = StatisticsTable(stats_df)
    liwc_spider = LiwcSpider(liwc_df) 

#GENRE DETECTION___________________________________________________________________________
    genre_detector = joblib.load('./stylene/genre_detector.sav')

    def predict_genre(text):
        process = spacy.load('nl_core_news_sm')
        text = ' '.join([t.text for t in process(text)])
        probs = genre_detector.predict_proba([text])[0]
        probs = [round(p,3) for p in probs]
        classes = genre_detector.classes_
        data = {k:[v] for k,v in zip(classes, probs)}
        df = pd.DataFrame(data=data)
        return df

    def GenreSpider(df):
        fig = px.line_polar(df, r=df.iloc[0].to_list(), theta=df.columns, line_close=True, title='Tekstgenre', color_discrete_sequence=['rgb(0,46,101)'])
        fig.update_traces(fill='toself')
        fig.update_layout(title_text='Figuur 3: Voorspelde kans per tekstgenre.', title_x=0.5, title_font=dict(size=12), title_font_family='Trebuchet MS', font_family='Trebuchet MS', font=dict(color='rgb(0,46,101)'))
        return fig

    genre_df = predict_genre(text)
    genre_spider = GenreSpider(genre_df)

#AUTHORSHIP ATTRIBUTES______________________________________________________________________
    education_pipe = joblib.load('./stylene/education_classifier.sav')
    age_pipe = joblib.load('./stylene/age_classifier.sav')
    gender_pipe = joblib.load('./stylene/gender_classifier.sav')
    personality_pipe = joblib.load('./stylene/personality_classifier.sav')
    namedict = {'hoog': 'hoog opgeleid', 'laag': 'laag opgeleid'}

    # PREDICTION____________________________________________________________________________
    def predict(df, pipe):

        """Takes feature df and pipeline as input.
        Returns dataframe with classes as columns and p as values."""
        probs = pipe.predict_proba(df)[0]
        probs = [p for p in probs]
        classes = pipe.classes_
        if pipe == education_pipe:
            output = {namedict[k]: [round(v, 3)] for k,v in zip(classes, probs)}
        else:
            output = {k:[round(v, 3)] for k,v in zip(classes, probs)}
        df_out = pd.DataFrame(data=output)
        return df_out

    # VISUALIZATION FOR GENDER, EDUCATION, PERSONALITY______________________________________
    def AuthorBar(name, df):

        """Takes task name and df with predictions as input.
        Returns bar chart with results.
        (Use AgeBar for results of age classification)"""

        color_dict = {
            'man': '#002e65',
            'vrouw': 'white',
            'introvert': '#002e65',
            'extravert': 'white',
            'hoog opgeleid': '#002e65',
            'laag opgeleid': 'white',
        }
    
        fig = go.Figure()
        classes = df.columns.to_list()

        for cl in classes:

            fig.add_trace(go.Bar(
                y=[name],
                x=[df.at[0, cl]],
                orientation='h',
                width=.75,
                marker=dict(
                    color=color_dict[cl],
                    line=dict(color='#002e65', width=3)
                ),
                text=cl+f' ({round(df.at[0, cl]*100, 3)}%)',
                textposition='inside',
                insidetextanchor='middle',
                showlegend=False,
            ))

            fig.update_yaxes(showticklabels=False)
            fig.update_xaxes(showticklabels=False)

            fig.update_layout(
                title_x=0.05,
                title_font=dict(size=12), 
                title_font_family='Trebuchet MS', 
                barmode='stack',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=100,
                margin=dict(l=40, r=40, t=50, b=0),
                font=dict(color='white'),
                font_family='Trebuchet MS', 
                )
            
            if name=='gender':
                fig.update_layout(
                    title_text='Figuur 1: Voorspelde auteurskenmerken.', 
                    title_x=0.5, 
                    title_font=dict(size=12, color='#002e65'), 
                    title_font_family='Trebuchet MS',
                    )

        return fig

    # VISUALIZATION FOR AGE DETECTION___________________________________________________________
    def AgeBar(df):

        """Takes df with age predictions as input.
        Returns timeline/bar chart with results as output."""
    
        fig = go.Figure()
        classes = df.columns.to_list()
        prediction = df.idxmax(axis=1).to_list()[0]

        for cl in classes:

            if cl == prediction:
                clr = '#002e65'
            else:
                clr = '#E5ECF6'

            if cl == '-18':
                x = 18
            elif cl == '18-25':
                x = 7
            else: 
                x = 25
            
            fig.add_trace(go.Bar(
                x=[x],
                orientation='h',
                width=1,
                marker=dict(
                    color=clr,
                    line=dict(color='#E5ECF6')
                ),
                insidetextanchor='middle',
                showlegend=False,
                hoverinfo='skip'
            ))

        fig.update_yaxes(showticklabels=False)

        fig.update_layout(
            title_x=0.05,
            title_font=dict(size=12), 
            title_font_family='Trebuchet MS', 
            barmode='stack',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=100,
            margin=dict(l=40, r=40, t=50, b=25),
            xaxis_title='leeftijd',
            font_family='Trebuchet MS',
            font=dict(color='#002e65'),
            xaxis=dict(tickmode='array', ticktext=[0, 18, 25, 50, 75])

            )

        return fig 
    
    #GENERATE DFs AND VISUALIZATIONS________________________________________________________________________________________
    gender_df = predict(feature_df, gender_pipe)
    education_df = predict(feature_df, education_pipe)
    age_df = predict(feature_df, age_pipe)
    personality_df = predict(feature_df, personality_pipe)

    gender_bar = AuthorBar('gender', gender_df)
    age_bar = AgeBar(age_df)
    personality_bar = AuthorBar('persoonlijkheid', personality_df)
    education_bar = AuthorBar('opleidingsniveau', education_df)

# AUTHOR STYLE COMPARISON____________________________________________________________________________________________________
    author_comparison_pipe = joblib.load('./stylene/author_detector.sav')

    def AuthorPrediction(df):
        """Takes feature df as input.
        Returns author predictions."""
        probs = author_comparison_pipe.predict_proba(df)[0]
        probs = [p for p in probs]
        classes = author_comparison_pipe.classes_
        output = {k.title():v for k,v in zip(classes, probs)}
        return pd.DataFrame(data={k: [v] for k,v in output.items()})
    
    def AuthorSpider(df):

        """Takes author predictions as input.
        Returns spider graph with results."""

        fig = px.line_polar(
            df, 
            r=df.iloc[0].to_list(), 
            theta=df.columns, 
            line_close=True, 
            title='Figuur 4: Stylistisch meest aansluitende literaire schrijver.', 
            color_discrete_sequence=['rgb(0,46,101)']
            )

        fig.update_traces(fill='toself')

        fig.update_layout(
            title='Figuur 4: Stylistisch meest aansluitende literaire schrijver.', 
            title_x=0.5, 
            title_font=dict(size=12), 
            title_font_family='Trebuchet MS', 
            font_family='Trebuchet MS', 
            font=dict(color='rgb(0,46,101)')
            )

        return fig
    
    author_prediction_df = AuthorPrediction(feature_df)
    author_spider = AuthorSpider(author_prediction_df)

#RETURN VISUALIZATIONS__________________________________________________________________________
    return {
        'gender_bar': gender_bar,
        'age_bar': age_bar,
        'personality_bar': personality_bar,
        'education_bar': education_bar,
        'genre_spider': genre_spider,
        'author_spider': author_spider,
        'liwc_spider': liwc_spider,
        'statistics_table': statistics_table,
        'pos_table': pos_table,
        'punct_table': punct_table,
        }
