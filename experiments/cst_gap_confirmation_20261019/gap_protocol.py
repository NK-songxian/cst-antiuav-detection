"""Causal history policy: exact offsets, identical boundary fallback to the old study."""
from collections import defaultdict

def build_history(index,far):
    assert far in (10,15,20)
    groups=defaultdict(dict)
    for stem,r in index.items():
        assert r['frame'] not in groups[r['seq']], 'duplicate sequence/frame'
        groups[r['seq']][r['frame']]=stem
    result={}
    for stem,r in index.items():
        lookup=groups[r['seq']]
        near=lookup.get(r['frame']-5,stem)
        distant=lookup.get(r['frame']-far,near)
        result[stem]=[distant,near,stem]  # Disk BGR; network RGB is current,near,far.
    return result
