from __future__ import division, print_function

import csv

from sklearn.utils import check_random_state
import numpy as np  

import match_utils as mu 

def read_jslda_doctopic_file(doctopic_file):

	para_to_topics = {} # a paragraph is a (speechname, paranum) tuple

	with open(doctopic_file, 'rb') as f:

		reader = csv.reader(f)
		for row in reader:
			name = row[0].split('_')

			speechname = '_'.join(name[:-1])
			paranum = int(name[-1])

			if row[1] == 'NaN':
				topic_weights = None
			else:
				topic_weights = np.array([float(x) for x in row[1:]])

			para_to_topics[(speechname, paranum)] = topic_weights

	return para_to_topics

def read_jslda_topic_file(topicsummary_file):

	summaries = []

	with open(topicsummary_file, 'rb') as f:

		reader = csv.reader(f)
		headings = reader.next()

		for row in reader:

			summaries.append(row[-1])

	return summaries



	
