import sys, os

from subprocess import run, PIPE

from django.shortcuts import render, redirect
from django.http import Http404

from .forms import ReaderForm
from .models import Reader
from .stylene import stylene

import plotly.express as px
from plotly.offline import plot


def index(request): 
	"""Home page""" 
	return render(request, 'index.html') 

def handler404(request, *args, **argv):
    response = render_to_response('404.html', {},
                                  context_instance=RequestContext(request))
    response.status_code = 404
    return response

def handler500(request, *args, **argv):
    response = render_to_response('500.html', {},
                                  context_instance=RequestContext(request))
    response.status_code = 500
    return response

def results(request):

	"""Process data and show results"""
	if "run_script" in request.POST:
		if request.FILES:
			file = request.FILES['data']
			inpt = ''
			for l in file:
				inpt += l.decode().strip()
		else:
			inpt = request.POST.get('data')

		output = stylene(inpt)
		
		gender_chart = output['gender_bar'].to_html()
		age_chart = output['age_bar'].to_html()
		education_chart = output['education_bar'].to_html()
		personality_chart = output['personality_bar'].to_html()
		liwc_chart = output['liwc_spider'].to_html()
		genre_chart = output['genre_spider'].to_html()
		author_chart = output['author_spider'].to_html()
		statistics_table = output['statistics_table'].to_html()
		pos_table = output['pos_table'].to_html()
		punct_table = output['punct_table'].to_html()

		context = {
			'gender_chart': gender_chart,
			'age_chart': age_chart,
			'education_chart': education_chart,
			'personality_chart': personality_chart,
			'liwc_chart': liwc_chart,
			'genre_chart': genre_chart,
			'statistics_table': statistics_table,
			'pos_table': pos_table,
			'punct_table': punct_table,
			'author_chart': author_chart
			}

		return render(request, 'results.html', context)

	else:
		return render(request, 'home.html')