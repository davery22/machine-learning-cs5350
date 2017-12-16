import re
from dateutil.parser import parse

def main():
    hashtag = r'(^|\s+)#\w'
    profanity = r'(fuck|shit|ass|dick|cunt|bitch|crap|butt)' # remember this is for science...
    results = []
    ids = []
    mapping = {}
    with open('./data/data-splits/data.train', 'r') as trainfile:
        for line in trainfile.readlines():
            results.append(line[0])

    with open('./data/data-splits/data.train.id', 'r') as idfile:
        for i,line in enumerate(idfile.readlines()):
            idd = line.strip()
            ids.append(idd)
            mapping[idd] = {'i':i, 'label':results[i], 'count_profanity':0.0, 'count_hashtags':0.0, 'count_messages':0.0, 'timestamps':[], 't_mean':0.0, 't_stddev':0.0}

    with open('./data/raw-data/tweets.txt', 'r') as rawfile:
        for line in rawfile.readlines():
            line = line.strip().split('\t')
            idd = line[0]
            if not idd in mapping:
                continue
            line2 = ''.join(line[1:])
            mapping[idd]['count_messages'] += 1
            mapping[idd]['count_hashtags'] += len(re.findall(hashtag, line2))
            mapping[idd]['count_profanity'] += len(re.findall(profanity, line2))
            mapping[idd]['timestamps'].append(parse(line[-1]))

    mapping = {k:v for k,v in mapping.iteritems() if v['count_messages']}
    avg_prof = 0.0#sum([v['count_profanity'] for k,v in mapping.iteritems()]) / len(mapping)
    avg_hasht = 0.0#sum([v['count_hashtags'] for k,v in mapping.iteritems()]) / len(mapping)

    for idd,dic in mapping.iteritems():
        if len(dic['timestamps']) < 2:
            continue
        for i in range(len(dic['timestamps'])-1):
            dic['timestamps'][i] = abs((dic['timestamps'][i+1] - dic['timestamps'][i]).total_seconds())
        dic['timestamps'] = dic['timestamps'][:-1]
        # Mean
        dic['t_mean'] = sum(dic['timestamps']) / float(len(dic['timestamps']))
        if len(dic['timestamps']) < 2:
            continue
        dic['t_stddev'] = (sum(map(lambda x: (x - dic['t_mean']) ** 2, dic['timestamps'])) / (len(dic['timestamps'])-1)) ** 0.5

    results = []
    for idd in ids:
        prof = mapping[idd]['count_profanity']/mapping[idd]['count_messages'] if mapping.has_key(idd) else avg_prof
        hasht = mapping[idd]['count_hashtags']/mapping[idd]['count_messages'] if mapping.has_key(idd) else avg_hasht
        tmean = mapping[idd]['t_mean'] if mapping.has_key(idd) else 0.0
        tstddev = mapping[idd]['t_stddev'] if mapping.has_key(idd) else 0.0
        results.append(' 17:{} 18:{} 19:{} 20:{}\n'.format(prof, hasht, tmean, tstddev))

    with open('./data/data-splits/data.new-train', 'w') as newtrain:
        with open('./data/data-splits/data.train', 'r') as oldtrain:
            for i,line in enumerate(oldtrain.readlines()):
                newtrain.write(line.strip() + results[i])

if __name__ == '__main__':
    main()

