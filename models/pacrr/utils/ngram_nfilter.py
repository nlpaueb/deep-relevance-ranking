# Original code authors: Kai Hui and Andrew Yates
# This is a copy of the code of Kai Hui and Andrew Yates, accompanying the following papers:
# Kai Hui, Andrew Yates, Klaus Berberich, Gerard de Melo. PACRR: A Position-Aware Neural IR Model for Relevance Matching. In EMNLP, 2017.
# Kai Hui, Andrew Yates, Klaus Berberich, Gerard de Melo. Co-PACRR: A Context-Aware Neural IR Model for Ad-hoc Retrieval. In WSDM, 2018.
# Github repository of the original code: https://github.com/khui/copacrr

def _get_filter_name(nx, ny):
    return '%dx%d'%(nx, ny)

def add_2dfilters(NGRAM_NFILTER=dict(), filters2add=list()):
    for nx2a, ny2a in filters2add:
        if ny2a not in NGRAM_NFILTER:
            NGRAM_NFILTER[ny2a]=list()
        filternames = set()
        for nx, ny in NGRAM_NFILTER[ny2a]:
            filtername = _get_filter_name(nx, ny)
            filternames.add(filtername)
        filtername = _get_filter_name(nx2a, ny2a)
        if filtername not in filternames:
            NGRAM_NFILTER[ny2a].append((nx2a, ny2a))

def add_ngram_nfilter(NGRAM_NFILTER=dict(), max_ngram=5):
    filters2add=list()
    for ngram in range(1, max_ngram+1):
        filters2add.append((ngram, ngram))
    add_2dfilters(NGRAM_NFILTER=NGRAM_NFILTER, filters2add=filters2add)

def add_proximity_filters(NGRAM_NFILTER=dict(), proximity=0, len_query=16):
    if proximity > 0:
        add_2dfilters(NGRAM_NFILTER=NGRAM_NFILTER, filters2add=[(len_query, proximity)])

def parse_more_filter(more_filters):
    '''
    convert the input string to a list of 2d filter sizes in form of tuple.
    input format: axb,cxd,...
    for example: 1x100,3x1 => [(1,100), (3,1)]
    '''
    filtersadd = list()
    if len(more_filters)==0:
        return filtersadd
    tups = more_filters.split('.')
    for tup in tups:
        a_b = tup.split('x')
        if len(a_b) != 2:
            raise ValueError("malformed of the input filter sizes %s"%more_filters)
        filtersadd.append((int(a_b[0]), int(a_b[1])))
    return filtersadd

def get_ngram_nfilter(max_ngram, proximity, len_query, more_filter_str):
    NGRAM_NFILTER=dict()
    add_ngram_nfilter(NGRAM_NFILTER=NGRAM_NFILTER, max_ngram=max_ngram)
    add_proximity_filters(NGRAM_NFILTER=NGRAM_NFILTER, proximity=proximity, len_query=len_query)
    filters2add = parse_more_filter(more_filter_str)
    add_2dfilters(NGRAM_NFILTER=NGRAM_NFILTER, filters2add=filters2add)
    N_GRAMS = list(NGRAM_NFILTER.keys())
    if len(N_GRAMS) == 0:
        N_GRAMS.append(0)
    return NGRAM_NFILTER, N_GRAMS
