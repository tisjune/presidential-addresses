from quote_searcher import QuoteSearcher  
import os

TRANSCRIPT_DIR = "/NLP/creativity/work/pres_addrs/transcripts/"
spinn3r_dir = "/NLP/creativity/nobackup/extractor_run/"

qs = QuoteSearcher(TRANSCRIPT_DIR, "dumps", tolerance=-.4, bag_tolerance = 0.7)
filelist = [os.path.join(spinn3r_dir, f) for f in os.listdir(spinn3r_dir) if f.endswith('.gz')]
c1 = 0
for f in filelist:
	cm = False
	if (c1%1000 == 0):
		cm = True
	c1 +=1
	qs.read_file(f, reread=False, cache_matches=cm)
	if cm:
		qs.dump_mentions()
	print "found "+str(len(qs.mentions))
	print "read " + str(len(qs.read_files)) + " files"
	count = len(qs.read_files)
	with open("log/"+str(count)+".txt", 'w') as g:
		g.write("found "+str(len(qs.mentions)) + " mentions up to file " + f)
qs.dump_mentions()
