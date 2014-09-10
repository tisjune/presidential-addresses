from __future__ import division
import group_quotes as gq 
import article_utils as au 
import match_utils as mu
import cPickle

'''
	groups quotes into families of similar quotes and deduplicates articles.
'''

TRANSCRIPT_DIR = '/NLP/creativity/work/pres_addrs/transcripts' # transcripts
POSTFILTERED_MENTION_OUTPUT = 'output/postfiltered_mentions' # output (pickle) of postfilter_matches.py
POSTPROCESSED_OUTPUT = 'output/postprocessed_mentions' # output file (pickle)

print 'loading stuff'
with open(POSTFILTERED_MENTION_OUTPUT) as f:
	mentions = cPickle.load(f)
torder, ttext = mu.fetch_transcripts(TRANSCRIPT_DIR)

print 'clustering quotes'
quotes = list(set([au.get_alignment(x) for x in mentions]))

transcript_to_quotelist = gq.get_transcript_to_quotelist(quotes)

alignment_to_family_id, family_id_to_alignments = gq.group_all(transcript_to_quotelist, 
			[x[1] for x in torder])
print len(family_id_to_alignments)
print 'deduplicating stuff'
url_to_article = au.simple_dedup(mentions)
families_to_articles = au.group_articles_by_cluster(mentions, url_to_article, alignment_to_family_id)
article_groups = []
for famset, articles in families_to_articles.iteritems():
	article_groups += au.get_article_groups(articles)
print len(article_groups)
url_to_article = au.load_complex_dedup_results(url_to_article, article_groups)
print len(set(url_to_article.values()))
print 'postprocessing'
postprocessed_mentions = []
# replace mentions with correct representative article and quote group
for m in mentions:
	alignment = au.get_alignment(m)
	family_id = alignment_to_family_id[alignment]
	raw_article = au.get_article(m)
	raw_url = raw_article[1]
	rep_article = url_to_article[raw_url]
	postprocessed_mentions.append((family_id, alignment, rep_article))

postprocessed_mentions = list(set(postprocessed_mentions))
print 'new length:'
print len(postprocessed_mentions)

with open(POSTPROCESSED_OUTPUT,'w') as f:
	cPickle.dump(postprocessed_mentions, f)
