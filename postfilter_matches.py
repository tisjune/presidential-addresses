from __future__ import division
import cPickle
import tldextract

'''
	gets rid of not real matches and wn.com articles. it works ok. 
'''

RAW_MENTION_OUTPUT = 'output/raw_mentions' #unfiltered mentions, a pickle
POSTFILTER_OUTPUT = 'output/postfiltered_mentions' #where to write filtered mentions, also a pickle

def short_seg_filter(segs, thresh = .5): 
	'''
		keeps quote (i.e. returns True) if for all segments, at least proportion=threshold
		of words are actually in the transcript it was matched to.
	'''
    keep = True
    for x in segs:
        if sum([y==-1 for y in x])/len(x) >= thresh:
            keep = False
    return keep

def short_quote_filter(segs, score, thresh_len = 6):
	'''
		filters out short quotes which do not match perfectly.
	'''
    if len(segs) > 1:
        return True
    else:
        seg = segs[0]
        if len(seg)>thresh_len:
            return True
        else:
            if sum([y >= 0 for y in seg]) == len(seg):
                return True
            else:
                return score >= 0

def score_filter(segs, score, thresh = -.37):
	'''
		filters out low-similarity alignments.
	'''
    if len(segs) > 1:
        return True
    else:
        return score > thresh


def ratio_filter(segs, thresh = .3, small_thresh = 7):
	'''
		removes quote if a large proportion of words aren't actually
		present in transcript.
	'''
    if len(segs) > 1:
        return True
    else:
        ratio = sum([y==-1 for y in segs[0]])/len(segs[0])
        if ratio > thresh:
            return False
        else:
            if len(segs[0]) > small_thresh:
                return True
            else:
                return ratio == 0



with open(RAW_MENTION_OUTPUT) as f:
    mentions = cPickle.load(f)

filtered_by_segments = [x for x in mentions if short_seg_filter(x[1][0])]
filtered_by_perfshorts = [x for x in filtered_by_segments if short_quote_filter(x[1][0],x[1][2])]
filtered_for_score = [x for x in filtered_by_perfshorts if score_filter(x[1][0],x[1][2])]
filtered_for_ratio = [x for x in filtered_for_score if ratio_filter(x[1][0])]
filtered_for_wn = [x for x in filtered_for_ratio if tldextract.extract(x[2]['url']).domain != 'wn']


with open(POSTFILTER_OUTPUT,'w') as f:
	cPickle.dump(filtered_for_wn, f)
