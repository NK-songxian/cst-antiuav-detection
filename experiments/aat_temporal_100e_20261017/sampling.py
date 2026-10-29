"""Exact primary-example budgets, rotating omissions, and causal image-only context."""
import random
from collections import defaultdict


def draw(tc, other, desired, rng):
    assert len(tc) <= desired <= min(2*len(tc),len(tc)+len(other))
    return list(tc)+rng.sample(tc, desired-len(tc))+rng.sample(other,len(tc)+len(other)-desired)


def build_schedule(index, mode, seed, epoch):
    rng=random.Random(seed+1000003*epoch)
    negatives=[]
    groups=defaultdict(list)
    for stem, rec in sorted(index.items()):
        if rec['exist']:
            groups[rec['seq']].append(stem)
        else:
            negatives.append(stem)
    if mode=='uniform':
        result=sorted(index)
    else:
        quota=[]
        tc_all=[]
        other_all=[]
        for seq, stems in sorted(groups.items()):
            tc=[s for s in stems if index[s]['attr'].get('TC')==1]
            other=[s for s in stems if index[s]['attr'].get('TC')!=1]
            desired=round(len(stems)*1.5*len(tc)/(1.5*len(tc)+len(other)))
            desired=max(len(tc),min(desired,2*len(tc),len(stems)))
            quota.extend(draw(tc,other,desired,rng))
            tc_all.extend(tc)
            other_all.extend(other)
        dose=sum(index[s]['attr'].get('TC')==1 for s in quota)
        if mode=='quota':
            result=quota+negatives
        elif mode=='global':
            result=draw(tc_all,other_all,dose,random.Random(seed+1000003*epoch))+negatives
        else:
            raise ValueError(mode)
    rng.shuffle(result)
    assert len(result)==len(index)
    return result


def build_history(index):
    """Use only exact t-10/t-5 matches in this split; clamp missing history to earlier available context."""
    groups=defaultdict(dict)
    for stem, rec in index.items():
        groups[rec['seq']][rec['frame']]=stem
    result={}
    for stem, rec in index.items():
        lookup=groups[rec['seq']]
        frame=rec['frame']
        previous=lookup.get(frame-5,stem)
        previous2=lookup.get(frame-10,previous)
        result[stem]=[previous2,previous,stem]
    return result
