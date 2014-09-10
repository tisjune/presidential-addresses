from collections import defaultdict

'''
	some utility functions for grouping quotes into families.
	Grouping quotes works as follows:
		1. for each transcript, quotes aligned to that transcript
			are sorted in order of their position in the transcript.
		2. for each quote, we see how many words overlap between that quote
			and the previous one.
		3. if the overlap is above some minimum threshold, we put that quote in the same group.
			else we form a new quote group.
 '''



def get_first_pos(seq): 
	'''
		gets the first positive value in a sequence.
	'''
	first_pos = seq[0]
	index = 0
	while (first_pos < 0):
		first_pos = seq[index]
		index += 1
	return first_pos

def group_quotes(quotelist, family_start_id, min_overlap = 5):
	'''
		groups a list of quotes aligned to the same transcript.

		arguments:
			quotelist (list of alignments): list of quotes, represented as alignments.
			family_start_id (int): the group id to start counting from
			min_overlap (int, default=5): minimum overlap between two quotes such that
				they are put in the same group.
		returns:
			family_assigns (list of int): list of quote group ids, in order of quotes listed
				in quotelist argument
			curr_id (int): last group id assigned to a quote.
	'''
    family_assigns = []
    curr_id = family_start_id
    prevquote = quotelist[0]
    extent = max(prevquote[-1]) #we (somewhat arbitrarily) suppose the quote is represented by its last segment.
    for q in quotelist:
        overlen = sum([x <= extent and x >= 0 for x in q[-1]])
        if overlen < min_overlap:
            curr_id += 1
        family_assigns.append(curr_id)
        prevquote = q
        extent = max(extent, max(q[-1]))
    return family_assigns, curr_id

def get_transcript_to_quotelist(alignment_list):
	'''
		returns a dict of transcript name to list of quotes aligned to that transcript
	'''
	transcript_to_quotelist = defaultdict(set)
	for q in alignment_list:
		transcript = q[1]
		quote = q[0]
		transcript_to_quotelist[transcript].add(quote)
	for transcript, quotelist in transcript_to_quotelist.iteritems():
		transcript_to_quotelist[transcript] = sorted(quotelist, 
			key=lambda x: get_first_pos(x[-1]))
	return transcript_to_quotelist

def group_all(transcript_to_quotelist, transcript_order, min_overlap = 5, start_id = 0):
	'''
		groups quotes into families of similar quotes.
		
		arguments:
			transcript_to_quotelist (dict): output of get_transcript_to_quotelist
			transcript_order: list of transcript names in order of increasing timestamp
			min_overlap
			start_id (int): the value of the first group id to be assigned.
		returns:
			alignment_to_family_id: dict of quote alignments (as tuple(index, transcriptname)) 
				to group id
			family_id_to_alignments: dict of group id to set of quotes, represented as alignments,
				in that group.
	'''
    alignment_to_family_id = {}
    family_id_to_alignments = defaultdict(set)
    f_id = start_id
    for transcript in transcript_order:
        quotelist = transcript_to_quotelist[transcript]
        if len(quotelist) == 0:
            continue
        f_assigns, curr_id = group_quotes(quotelist, f_id)
        for quote, assign in zip(quotelist, f_assigns):
            alignment_to_family_id[(quote, transcript)] = assign
            family_id_to_alignments[assign].add((quote, transcript))
        f_id = curr_id + 1
    return alignment_to_family_id, family_id_to_alignments
