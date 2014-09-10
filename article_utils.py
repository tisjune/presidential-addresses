from __future__ import division
import datetime as dt 
from sets import ImmutableSet
from collections import defaultdict
from fuzzywuzzy import fuzz
import tldextract
import re

'''
	handles deduplication of articles; also finds and handles AP and Reuters wire articles.

	Note that throughout these routines, we use the URL present in the original mention (as outputted
		by quote_searcher.py) to refer to a mention, and map the article in that mention to the 
		representative article returned by deduplication.
'''

AP_COPYRIGHT_RE = r'copyright \d{4} the associated press'
AP_URL = 'http://ap.org/'

def get_alignment(mention_entry):
	'''
		from a mention entry (as outputted by quote_searcher.py) returns the quote alignment.
	'''
	aligned_indices = tuple(mention_entry[1][0])
	transcript_name = mention_entry[1][1]
	return (aligned_indices, transcript_name)

def get_article(mention_entry):
	'''
		from a mention entry (as outputted by quote_searcher.py) returns the important information
		in an article as a tuple.
	'''
	article_dict = mention_entry[2]
	title = article_dict['title']
	url = article_dict['url']
	timestamp = article_dict['date']
	content = article_dict['content']
	quotes = tuple(article_dict['quotes'])
	return (title, url, timestamp, content, quotes)

def simple_dedup(mentions):
	'''
		given a list of mentions, deduplicates based on URL and title.
		returns a dict of URL (original URL in each mention entry) to representative article 
			(chosen to be the earliest article with the same URL/title).
	'''
	url_to_article = {}
	for m in mentions:
		article = get_article(m)
		url = article[1]
		timestamp = article[2]

		rep_article = url_to_article.get(url, None)
		if rep_article is None or timestamp < rep_article[2]:
			url_to_article[url] = article

	# probably equivalent and faster to just combine the two loops.
	# but not probable enough that I'm going to do it.

	title_to_article = {}
	for article in url_to_article.values():
		title = article[0]
		timestamp = article[2]
		rep_article = title_to_article.get(title, None)
		if rep_article is None or timestamp < rep_article[2]:
			title_to_article[title] = article

	for url, article in url_to_article.iteritems():
		title = article[0]
		url_to_article[url] = title_to_article[title]

	return url_to_article


def group_articles_by_cluster(mentions, url_to_article, alignment_to_fid):
	'''
		groups articles according to the quote groups present in them. (where
			quote groups are as outputted by group_quotes.py)
	'''
	url_to_families = defaultdict(set)
	for m in mentions:
		article = get_article(m)
		url = article[1]
		alignment = get_alignment(m)
		fid = alignment_to_fid[alignment]
		rep_url = url_to_article[url][1]
		url_to_families[rep_url].add(fid)
	for url, famset in url_to_families.iteritems():
		url_to_families[url] = ImmutableSet(famset)

	families_to_articles = defaultdict(list)
	for url, famset in url_to_families.iteritems():
		rep_article = url_to_article[url]
		families_to_articles[famset].append(rep_article)
	for famset, articles in families_to_articles.iteritems():
		families_to_articles[famset] = sorted(articles, key=lambda x: x[2])

	return families_to_articles

def get_article_groups(article_list, fuzz_len=3000, min_ratio = 70):
	'''
		finds duplicate articles in a list of articles.
		
		arguments:
			article_list: list of articles
			fuzz_len (int, default=3000): number of characters at 
				beginning/end of article to compare against using fuzz
			min_ratio (int, default=70): minimum fuzz score (as percentage)
				above which two articles are considered duplicates.

		returns:
			list of ints for each article in article_list, where each distinct int
				corresponds to a distinct group of duplicate articles.
		
	'''
	if len(article_list) == 1:
		return [article_list]
	else:
		rep_articles = []
		rep_ids = []
		rep_num = -1
		fuzzratio = 0
		for article in article_list:
			a_content = article[3]
			a_len = len(a_content)
			matched = False
			for i in range(len(rep_articles)):
				rep_content = rep_articles[i][3]
				rep_len = len(rep_content)
				minlen = min(fuzz_len, a_len, rep_len)
				fuzzratio = fuzz.ratio(rep_content[:minlen], a_content[:minlen])
			# we compare the beginnings and ends of both articles using fuzz. 
			# it's possible that some of these checks are mostly useless; this is slow.
			# but for deduplication I've chosen to be careful.
				if fuzzratio < min_ratio:
					fuzzratio = fuzz.ratio(rep_content[-minlen:],a_content[-minlen:])
				if fuzzratio < min_ratio:
					fuzzratio = fuzz.ratio(rep_content[-minlen:],a_content[:minlen])
				if fuzzratio < min_ratio:
					fuzzratio = fuzz.ratio(a_content[-minlen:],rep_content[:minlen])
				if fuzzratio >= min_ratio:
					rep_ids.append(i)
					matched = True
 					break
			if not matched:
				rep_num += 1
				rep_ids.append(rep_num)
				rep_articles.append(article)
		groups = []
		# assign duplicate article groups to each article
		for i in range(len(rep_articles)):
			groups.append([article_list[j] for j in range(len(article_list)) if rep_ids[j] == i])
		return groups


# tries rather hard to id ap and reuters stories. returns 'ap' and 'reuters' in those cases and None else
# theoretically, since spinn3r seems to pull reuters stories properly, we don't have to try as hard
def get_wire_story(article):
    
    title = article[0]
    url = article[1]
    content = article[3]
    
    domain = tldextract.extract(url).domain
    if domain == 'ap':
        return 'ap'
    elif domain == 'reuters':
        return 'reuters'
    
    if 'blog' in url or 'opinion' in url: 
        return None 
    
    if 'ap news' in title or 'associated press' in title:
        return 'ap'
    
    if 'reuters' in title:
        return 'reuters'
    
    if re.search(AP_COPYRIGHT_RE, content) is not None:
        return 'ap'
    
    if 'the associated press contributed to this' in content: # i have no idea what fox means when it says that
        return None
    

    if ('associated press' in content[:500] or '(ap)' in content[:500] 
            or 'associated press' in content[-500:] or '(ap)' in content[-500:]):
        return 'ap'

    if ('(reuters)' in content):
        return 'reuters'
    
    return None

def get_representative_article(article_list):
	'''
		for a list of duplicate articles, returns what we think is probably
			the earliest occuring article.
		Rules:
			1. if any of the articles are a wire source, return the wire source article.
			2. do not return articles from know aggregators such as free republic
			3. return the earliest article by timestamp.
	'''
	if len(article_list) == 1:
		wire_id = get_wire_story(article_list[0])
		return article_list[0], wire_id
	else:

		#check if anything is a wire story 
		for article in article_list:
			wire_id = get_wire_story(article)
			if wire_id is not None:
				return article, wire_id
		#return the earliest article, provided it's not from freerepublic
		for article in article_list:
			if tldextract.extract(article[1]).domain != 'freerepublic':
				return article, None

		#call it a day
		return article_list[0], None 

def load_complex_dedup_results(url_to_article, article_groups):
	'''	
		outputs the results of our more complicated article deduplication routine
			(in article_groups):

		returns a dict of 
			url (as present in the mention outputted by quote_searcher.py)
			to representative article.
	'''
	rep_url_to_rep_article = {}
	for group in article_groups:
		rep, wire_id = get_representative_article(group)
		if wire_id == 'ap':
			rep = (rep[0],AP_URL,rep[2], rep[3],rep[4])
		for article in group:
			url = article[1]
			rep_url_to_rep_article[url] = rep
	toreturn = {}
	for url, article in url_to_article.iteritems():
		rep_url = article[1]
		toreturn[url] = rep_url_to_rep_article[rep_url]
	return toreturn
