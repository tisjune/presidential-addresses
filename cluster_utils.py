from __future__ import division
import cPickle
import datetime as dt 
from collections import defaultdict
import tldextract
import numpy as np  
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

'''
	functions to cluster domains by quotation pattern.
'''

def load_mentions(mentions):
	'''
		reads list of mentions (as outputted by postprocess_matches.py) and returns:
			list of citations as (domain, id of quote group cited)
			dict of (domain, id) to representative article
			dict of quote group id to most frequent representative quote of that group
				(listed as a quote aligment, not raw text)
			
	'''	
	fid_to_counts = defaultdict(dict)
	dom_fid_to_article = {}

	dom_fid_set = set()
	for m in mentions:

		fid = m[0]
		alignment = m[1]
		article = m[2]

		domain = tldextract.extract(article[1]).domain

		count = fid_to_counts[fid].get(alignment, 0)
		fid_to_counts[fid][alignment] = count + 1

		timestamp = article[2]
		rep = dom_fid_to_article.get((domain, fid), None)
		if rep is None or timestamp < rep[0][2]:
			dom_fid_to_article[(domain, fid)] = (article, alignment)

		dom_fid_set.add((domain, fid))

	fid_freq_reps = {}
	for fid, countdict in fid_to_counts.iteritems():
		sorted_by_count = sorted(countdict.iteritems(), key=lambda x: x[1])
		fid_freq_reps[fid] = sorted_by_count[-1][0]

	return list(dom_fid_set), dom_fid_to_article, fid_freq_reps


def get_domain_fid_matrix(dom_fid, fid_set = None):
	'''
		given list of citations [(domain, quote group id)]
		produces binary matrix of domains to quotes, with values = 1 (domain cited quote)

		in binary matrix, domains ordered by frequency; quotes ordered by group id value.

		returns binary matrix, list of (domain, frequency) ordered by frequency,
			dict of domain name to index of domain in matrix.
	'''
	if fid_set is None:
		df_list = dom_fid
	else:
		df_list = [x for x in dom_fid if x[1] in fid_set]

	domain_counts = defaultdict(int)
	for dom, fid in df_list:
		domain_counts[dom] += 1

	top_domains = sorted(domain_counts.iteritems(), key=lambda x: x[1], reverse=True)

	num_domains = len(top_domains)

	domain_to_index = {top_domains[i][0]:i for i in range(len(top_domains))}

	num_fids = len(set([x[1] for x in dom_fid]))

	dom_fid_matrix = np.zeros((num_domains, num_fids))
	for dom, fid in df_list:
		dom_index = domain_to_index[dom]
		dom_fid_matrix[dom_index,fid] = 1

	if fid_set is None:
		toreturn = dom_fid_matrix
	else:
		toreturn = dom_fid_matrix[:,fid_set]

	return toreturn, np.array(top_domains), domain_to_index

def get_sim_matrix(bin_matrix):
	'''
		returns similarity matrix between domains.
	'''
	col_norm = bin_matrix / np.sqrt(bin_matrix.sum(axis=0))
	normed_by_row = col_norm / np.linalg.norm(col_norm, axis=1)[:,np.newaxis]

	sim = np.dot(normed_by_row, normed_by_row.T)
	return sim

def get_dist_matrix(bin_matrix):
	'''
		returns distance matrix between domains
	'''
	return 1-get_sim_matrix(bin_matrix)

def hier_cluster_and_display(dist_matrix, leaf_labels, colorthresh, to_cluster = 'all', m = 'complete', 
			imgsize = 25, fontsize=16):
	'''
		clusters domains using hierarchical clustering and displays dendrogram.
		arguments:
			dist_matrix : distance matrix between domains
			leaf_labels: list of domain names
			colorthresh: threshold to color dendrogram nodes
			to_cluster (list of ints, optional, default='all'):
				if 'all', clusters all domains
				else clusters only domains corresponding to indices in list
			m (default='complete'): method used in hierarchical clustering.
				'single' and 'average' also work; as in scipy.
			imgsize (default=25): size of image (imgsize,imgsize) of dendrogram to produce.
			fontsize (default=16): font size of dendrogram leaf labels.
		returns:
			result as outputted by scipy's hierarchical clustering.
	'''
	if to_cluster == 'all':
		cluster_indices = range(dist_matrix.shape[0])
	else:
		cluster_indices = to_cluster
	plt.figure(figsize=(imgsize,imgsize))
	result = hier_cluster(dist_matrix,cluster_indices,m)
	dendrogram(result,orientation='left',
		labels=leaf_labels[cluster_indices], color_threshold=colorthresh, leaf_font_size=fontsize)

	return result

def display_cluster_result(result, leaf_labels, colorthresh, to_cluster = 'all',
		imgsize = 25, fontsize = 16):
	'''
		displays dendrogram corresponding to a clustering result.
	'''
	if to_cluster == 'all':
		cluster_indices = range(len(leaf_labels))
	else:
		cluster_indices = to_cluster

	plt.figure(figsize=(imgsize,imgsize))
	dendrogram(result, orientation='left',labels=leaf_labels[cluster_indices], color_threshold=colorthresh,
			leaf_font_size=fontsize)
	

def hier_cluster(dist_matrix, to_cluster = 'all', m='complete'):
	'''
		uses hierarchical clustering with specified distance matrix.
	'''
	if to_cluster == 'all':
		cluster_indices = range(dist_matrix.shape[0])
	else:
		cluster_indices = to_cluster
	dist_submatrix = dist_matrix[cluster_indices,:][:,cluster_indices]
	result = linkage(dist_submatrix, method=m)
	return result

def convert_to_indices(domain_list, domain_to_index):
	'''
		converts a list of domain names in domain_list to a list of indices given by
			domain_to_index.
	'''
	index_list = []
	for d in domain_list:
		try:
			index_list.append(domain_to_index[d])
		except KeyError:
			print 'No domain '+d 
			return
	return index_list

def get_avg_indistance(dist_matrix, index_list):
	'''
		computes the average distance between all domains in index_list, where
			index list is a list of indices into the distance matrix.
	'''
	dist_submatrix = dist_matrix[index_list,:][:,index_list]
	dist_sum = sum(sum(dist_submatrix))
	num_items = len(index_list)
	return dist_sum / (num_items * (num_items-1))
