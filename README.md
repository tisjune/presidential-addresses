presidential-address-project
============================

work on reading/analyzing presidential addresses and media coverage, at MPI-SWS 2014

contains code to

0. extract articles from spinner
    (extractor)
1. align news quotes with words in presidential addresses
    (match_utils.py, match_quotes.py, quote_searcher.py, get_matches.py for matching)
    (postfilter_matches.py, group_quotes.py, article_utils.py, postprocess_matches.py for postprocessing)
2. cluster news domains according to quote history
    (cluster_utils.py)
3. predict news domain quotes according to quote history
    (predict_quotes.py)

