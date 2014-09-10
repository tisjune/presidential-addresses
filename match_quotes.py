from __future__ import division
import datetime as dt 
import os, string, collections, cPickle, bisect
import match_utils as mu 


''' aligns quotes to a set of texts. 
'''

class QuoteMatcher:
	'''
		Aligns quotes to a set of texts.
	'''

	PHRASEMATCH_CACHENAME = 'phrase_matches'

	MIN_LEN = 6 # minimum word count of quote
	MAX_INTERVAL = dt.timedelta(days=14) # max time interval between transcript and quote
	GAP_PEN = -1 # gap penalty for needleman-wunsch
	SUB_PEN = -1 # substitution penalty for needleman-wunsch
	ACCEPT_THRESHOLD = -0.1 

	def __init__(self, transcript_dir, cache_dir, sim_tolerance = -0.4, bag_tolerance = 0.7):
		'''
			Arguments:
				transcript_dir (str): directory containing speech transcripts.
					Transcripts should be of format:
						Transcript title
						Transcript timestamp
						Transcript text
				cache_dir (str): directory to write cache output
				sim_tolerance (float, default=-.4): minimum similarity of valid quote-transcript match
				bag_tolerance (float, default=.7): minimum proportion of words in quote that must be
					present in transcript for string alignment process to occur. 
		'''	
		self.sim_tolerance = sim_tolerance
		self.bag_tolerance = bag_tolerance

		self.cache_dir = cache_dir

		transcript_order, self.transcript_text = mu.fetch_transcripts(transcript_dir)

		self.transcript_times = [x[0] for x in transcript_order]
		self.transcript_names = [x[1] for x in transcript_order]

		self.phrasematch_cache = self.load_cached_dict(self.PHRASEMATCH_CACHENAME)

	def load_cached_dict(self, cache_filename):
		'''
			Loads a phrasematch cache from cache_filename, a pickle file.
		'''
		cache_path = os.path.join(self.cache_dir, cache_filename)

		if os.path.isfile(cache_path):
			print "Loading "+cache_filename
			with open(cache_path, 'rb') as f:
				return cPickle.load(f)
		else:
			return {}

	def dump_cached(self, cached, cache_filename):
		'''
			Dumps phrasematch cache to a pickle file.
		'''
		print "Dumping " + cache_filename
		if not os.path.exists(self.cache_dir):
			os.mkdir(self.cache_dir)
		with open(os.path.join(self.cache_dir, cache_filename), 'wb') as f:
			cPickle.dump(cached, f)
		print "Finished dumping " + cache_filename

	def dump_all(self):

		self.dump_cached(self.phrasematch_cache, self.PHRASEMATCH_CACHENAME)

	# return: (bool match, (list(list) alignment, string transcriptname, score)). the tup is None if no match found.
	def match_quote(self, quote, quote_timestamp):
		'''
			Aligns a quote to a set of transcripts.

			Arguments:
				quote (str): quote to match
				quote_timestamp (datetime): timestamp of quote

			Returns:
				bool: True if match found (i.e. quote similarity score above threshold), False else.
				tuple containing:
					alignment: list of list of ints, best alignment for quote
					transcript name (str): transcript filename that quote aligned to
					score (float): alignment score
				or None if bool==False.	
			
			The precise process works as follows:
				1. check if quote is above minimum word length
				2. find transcripts within maximum time interval that occur before quote
				3. check if there is already an alignment for a quote (or an entry saying the
					quote couldn't be aligned) in the cache
				4. check if a sufficiently high proportion of words in the quote are also 
					present in the transcript
				5. align the quote to the transcript, and save the alignment to the cache.		
		'''
		
		# check if quote is above min len
		quote_array = mu.convert_to_match_array(quote)

		trial_length = len(quote_array)

		if trial_length >= self.MIN_LEN:
			# search for transcripts within time window.
			latest_transcript_index = bisect.bisect_left(self.transcript_times, quote_timestamp) - 1
			earliest_transcript_index = bisect.bisect_left(self.transcript_times, quote_timestamp - self.MAX_INTERVAL)
			search_range = range(latest_transcript_index, earliest_transcript_index - 1, -1)

			if latest_transcript_index < 0 or earliest_transcript_index >= len(self.transcript_times):
				return (False, None)

			else:

				best_align = None
				best_transcript = None
				best_score = None

				for i in search_range:

					curr_transcript_name = self.transcript_names[i]

					# first see if the entire quote is already cached.
					cached_align = self.phrasematch_cache.get((quote, curr_transcript_name), None)
					if cached_align is not None:
						if cached_align[0]:
							if cached_align[1][1] > best_score:
								best_align, best_score = cached_align[1]
								best_transcript = curr_transcript_name
					else:
						# check if enough words are present in the transcript that we should even bother looking
						in_set_count = sum([(word in self.transcript_text[curr_transcript_name]['bag']) 
								for word in quote_array])
						if in_set_count / trial_length < self.bag_tolerance:
							self.phrasematch_cache[(quote, curr_transcript_name)] = (False, None)
						else:
							#now we actually have to work hard
							#so that the cache isn't overwhelmed with tiny segments, we won't cache segments.
							align = mu.align(quote, self.transcript_text[curr_transcript_name]['raw'],
								self.transcript_text[curr_transcript_name]['match'], self.SUB_PEN, self.GAP_PEN)
							if align[1] >= self.sim_tolerance:
								self.phrasematch_cache[(quote, curr_transcript_name)] = (True, align)
								if align[1] > best_score:
									best_align = align[0]
									best_score = align[1]
									best_transcript = curr_transcript_name
							else:
								self.phrasematch_cache[(quote, curr_transcript_name)] = (False, None)
							
				if best_score >= self.sim_tolerance:
					return (True, (best_align, best_transcript, best_score))
				else:
					return (False, None)

		else: 
			return False, None
