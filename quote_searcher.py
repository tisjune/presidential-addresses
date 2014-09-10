import gzip
import os
import datetime as dt  
from match_quotes import QuoteMatcher
import cPickle

'''
	searches for quotes matching transcripts in a spinn3r data directory, returning mentions.

'''

TRANSCRIPT_TIMEFORMAT = "%Y-%m-%d %H:%M"
NEWS_TIMEFORMAT = "%Y-%m-%d %H:%M:%S"
HYPHEN_TYPES = ["\xe2\x80\x94", " - ", " -- "] 

class QuoteSearcher:

	MENTION_DUMPNAME = "mentions"

	def __init__(self, transcript_dir, dump_dir, cache_dir = 'cache', tolerance = -0.4, bag_tolerance = 0.7):
		'''
			Arguments:
				transcript_dir (str): directory containing transcripts. Transcripts are of form:
					Transcript title
					Transcript timestamp (%Y-%m-%d %H:%M)
					Transcript text
				dump_dir (str): directory to dump matches found
				cache_dir (str): directory to dump cache of quote matches
				tolerance (float, optional, default=-.4): minimum similarity score for match
				bag_tolerance (float, optional, default=.7): minimum proportion of words
					in quote that must be present in a transcript for alignment process to continue
		'''
		self.qm = QuoteMatcher(transcript_dir, cache_dir, tolerance, bag_tolerance)
		self.dump_dir = dump_dir
		self.mentions = []
		self.read_files = set()
		self.errors = 0

	def read_file(self, filename, reread=False, cache_matches=True):
		'''
			Reads one spinn3r file, matching all quotes from that file to transcripts.
			Arguments:
				filename (str): spinn3r filename
				reread (bool, optional, default=False): if True, will reread a file we've already read.
					(this is mostly for development: when program crashes on a new file, we don't really
					want to reread old ones.)
				cache_matches (bool, optional, default=True): if True, will save to disk the cache of quote matches.
		'''
		if reread or filename not in self.read_files:
			print "Reading "+filename
			with gzip.open(filename, 'rb') as f:
				linecount = 0
				for line in f:
					article = self.read_article(line)
					self.match_quotes(article)
					linecount += 1
					#if linecount % 200 == 0:
					#	print "Read "+str(linecount) + " lines"
			self.read_files.add(filename)
			print "Finished reading. Caching matches..."
			if cache_matches:
				self.qm.dump_all()

	def read_article(self, line):
		'''
			Reads single article, returning article dict.
		'''
		article_dict = eval(line) # hmmm

		strdate = article_dict['date']
		article_dict['date'] = dt.datetime.strptime(strdate, NEWS_TIMEFORMAT)

		quotes = []

		raw_quote_list = article_dict['quotes']
		for elem in raw_quote_list:
			quotes.append(elem['quote'])

		article_dict['quotes'] = quotes 

		return article_dict

	def dump_mentions(self, dump_filename=None):
		'''
			Writes all matches found as pickle to dump_filename.
		'''
		if dump_filename is None:
			dump_filename = self.MENTION_DUMPNAME
		mentions_dump = os.path.join(self.dump_dir, dump_filename)
		print "Dumping mentions into "+mentions_dump
		if not os.path.exists(self.dump_dir):
			os.mkdir(self.dump_dir)
		with open(mentions_dump, 'wb') as f:
			cPickle.dump(self.mentions, f) 
		print "Finished dumping mentions"

	def match_quotes(self, article):
		'''
			matches all quotes in an article.
		'''
		timestamp = article['date']
		quotes = article['quotes']

		for quote in quotes:
			try:
				match_result = self.qm.match_quote(quote, timestamp)
				if match_result[0]:
					self.mentions.append((quote, match_result[1], article))
			except:
				print "Ack"
				self.errors +=1

