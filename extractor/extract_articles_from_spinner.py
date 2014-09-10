'''
	Extracts spinn3r articles. A modification of this :
	https://github.com/rjweiss/SNAP-tools/blob/master/snaprar.py

	DESCRIPTION

	This code will read spinn3r data, provided in .rar format,
	and output a subset of news articles from a specified list of news domains
	and optionally which contain a specified list of keywords or phrases. 

	USAGE 

	Assuming that the spinn3r data is in a directory structured as such:

		spinner-directory/web/YYYY-MM/web-YYYY-MM-DDTHH-00-00Z.rar

	the code can be run as

	python extract_articles_from_spinner.py spinner-directory START-DATE END-DATE NEWS-DOMAIN-FILE [FILTER-WORDS-FILE]

	Start and end dates are specified in format YYYY-MM-DD. 
	The code will retrieve everything in the range [START-DATE, END-DATE). 
	You can specify arbitrarily early/late start and end dates. 

	NEWS-DOMAIN-FILE contains a list of domain names, one per line. We've been using ./worlddomains.

	FILTER-WORDS-FILE contains a list of words or phrases, one per line. 
	A news article will only be outputted if its title+content contains each of the keywords listed. 
	We've been using ./filterwords.

	OUTPUT 

	For each spinn3r file read, one file is produced in the directory the script was run in, titled:
		web-YYYY-MM-DDTHH-00-00Z_news.gz

	Files are gzipped, and list one article per line. Articles are strings of the following format:

	{ 
	  'links': num_links, 
	  'title': article_title, 
	  'url': article_url, 
	  'content': article_content, 
	  'quotes':[{'onset':#,'quote':quote,'length':len(quote)}], 
	  'file':spinner_file_name, 
	  'date':YYYY-MM-DD HH:MM:SS
	}

'''

import tldextract
#import rarfile
import subprocess
import os 
import sys
import tempfile
import shutil
import gzip
import datetime as dt
from multiprocessing import Pool

import logging
logging.basicConfig()

NEWSDOMAINS = set([])
FILTERWORDS = set([])

def newitem(name):
    return {
        'file': name,
        'url': '__nourl__',
        'date': '__nodate__',
        'title': '__notitle__',
        'content': '__nocontent__',
        'links': 0,
        'quotes': [],
    }

def printitem(ditem):

    if ditem['content'] != '__nocontent__':
        ditem['content'] = str(ditem['content'])

    if ditem['links'] == 0:
        ditem['links'] = '__nolinks__'
    else:
        ditem['links'] = str(ditem['links'])
            
    if ditem['quotes'] == []:
        ditem['quotes'] = '__noquotes__'
    else:
        ditem['quotes'] = str(ditem['quotes'])

    outline = [
        ditem['file'],
        ditem['url'],
        ditem['date'],
        ditem['title'],
        ditem['content'],
        ditem['links'],
        ditem['quotes'],
    ]

    return '\t'.join(outline)

def domainfilter(urlstring, domains):
	try:
		url = tldextract.extract(urlstring)
		if url.domain in domains:
			return urlstring
		else:
			return False
	except:
		return False

def contentfilter(contentstring, titlestring, filterwords): #the content and title must contain all the words.
	article_string = (contentstring + titlestring).lower()
	has_filterwords = True
	for word in filterwords:
		has_filterwords = has_filterwords and (word in article_string)
	return has_filterwords

def process_file(snaptext, filename, domains, filterwords):
	outfile = []
	snapfile = open(snaptext, 'r')
	ditem = newitem(filename)

	for line in snapfile:
		cline = line.split('\n')[0]				
		words = cline.split('\t')

		if len(words) <= 1:

			if ditem['url'] != '__nourl__' and contentfilter(ditem['content'], ditem['title'], filterwords):
				outfile.append(ditem)


			ditem = newitem(filename)

			continue

		key = words[0]
		value = words[1]
			

		if key == 'U':
			urlfilter = domainfilter(value, domains)
			if urlfilter:
				ditem['url'] = value
			else:
				continue	
		elif key == 'D':
			ditem['date'] = value
		elif key == 'T':
			ditem['title'] = value
		elif key == 'C':
			ditem['content'] = value
		elif key == 'L':
			#print value
			ditem['links'] += 1
		elif key == 'Q':
			ditem['quotes'].append(
				{
					'onset': words[1], 
					'length': words[2], 
					'quote': words[3]
				})
		else:
			print '*** ERROR: unknown key'
			#sys.exit(1)

		
	return outfile
	
	snapfile.close()

def read_spinner_file(spinner_filename):
	print "Reading "+spinner_filename

	#rf = rarfile.RarFile(spinner_filename)
	try:
		tmp_dir = tempfile.mkdtemp()
		subprocess.call(['7z', 'e', '-o'+tmp_dir, spinner_filename])
		#rf.extractall(path=tmp_dir)
		basename = os.path.basename(spinner_filename)
		filename = basename[0:-3]+"txt"
		flatsnap = process_file(os.path.join(tmp_dir, filename), basename, NEWSDOMAINS, FILTERWORDS)
		archive_name = os.path.splitext(filename)[0] + "_news.gz"
		with gzip.open(archive_name, 'wb') as archive:
			for item in flatsnap:
				archive.write('%s\n' % item)
	finally:
		try:
			shutil.rmtree(tmp_dir)
		except OSError, e:
			print "Error removing file?!?"
			if e.errno != 2:
				raise

def filetest(path):
	with open(path) as f:
		writestr = path 
		basename = os.path.basename(path)[0:-3]+'zip'
		with gzip.open(os.path.splitext(basename)[0]+"_news.gz",'wb') as archive:
			archive.write(writestr)


def gen_file_names(target_dir, start_date, end_date):
	toreturn = []
	for dirname, dirnames, filenames in os.walk(target_dir):
		for filename in filenames:
			filedate = dt.datetime.strptime(filename, "web-%Y-%m-%dT%H-00-00Z.rar")
			if (filedate <= end_date and filedate >= start_date):
				toreturn.append(os.path.join(dirname, filename))
	return toreturn

def main():

	if len(sys.argv) < 5:
		print "Usage: python extract_articles_from_spinner.py <spinn3r directory> <start-date (format: YEAR-MM-DD)> <end-date> <news domain file> (optional)<filter words file>"
		exit(1)

	target_dir = os.path.join(sys.argv[1],"web")

	start_date = dt.datetime.strptime(sys.argv[2], "%Y-%m-%d")
	end_date = dt.datetime.strptime(sys.argv[3], "%Y-%m-%d")
	with open(sys.argv[4], 'r') as newssites:
		for line in newssites:
			NEWSDOMAINS.add(line.strip())

	if len(sys.argv) >= 6:
		with open(sys.argv[5], 'r')	as filterwordfile:
			for line in filterwordfile:
				FILTERWORDS.add(line.strip().lower())

	pool = Pool(processes=7)
	
	spinn3r_files = gen_file_names(target_dir, start_date, end_date)
	print spinn3r_files
	results = pool.map(read_spinner_file, spinn3r_files)
	#results = pool.map(filetest, spinn3r_files)
	#for spinner_file in os.listdir(target_dir):
	#	read_spinner_file(os.path.join(target_dir,spinner_file))
	pool.close()
	pool.join()



if __name__ == '__main__':
	main()
