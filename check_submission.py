"""Independent check of a submission against the rules stated in the brief."""
import sys
import numpy as np
import pandas as pd

H = W = 672
root, path = sys.argv[1], sys.argv[2]
te = pd.read_csv(f'{root}/test.csv')
sub = pd.read_csv(path, dtype={'mask_rle': str})
errs = []

if list(sub.columns) != ['sample_id', 'mask_rle']:
    errs.append(f'columns are {list(sub.columns)}, expected [sample_id, mask_rle]')
if len(sub) == 0:
    errs.append('empty table')
want = set(te.sample_id)
got = list(sub.sample_id)
if len(got) != len(set(got)):
    errs.append('duplicate sample_id')
if set(got) - want:
    errs.append(f'unknown sample_id: {list(set(got)-want)[:3]}')
if want - set(got):
    errs.append(f'missing sample_id: {list(want-set(got))[:3]}')

nempty = 0
areas = []
for sid, s in zip(sub.sample_id, sub.mask_rle):
    if not isinstance(s, str):
        errs.append(f'{sid}: non-string value')
        continue
    s = s.strip()
    if s == 'no-plume':
        nempty += 1
        continue
    v = s.split()
    if len(v) % 2:
        errs.append(f'{sid}: odd number of values')
        continue
    try:
        v = [int(x) for x in v]
    except ValueError:
        errs.append(f'{sid}: non-integer token')
        continue
    occ = np.zeros(H * W, np.uint8)
    ok = True
    for st, ln in zip(v[0::2], v[1::2]):
        if ln <= 0:
            errs.append(f'{sid}: non-positive length')
            ok = False
            break
        if st < 1 or st - 1 + ln > H * W:
            errs.append(f'{sid}: run outside grid')
            ok = False
            break
        seg = occ[st - 1:st - 1 + ln]
        if seg.any():
            errs.append(f'{sid}: overlapping runs')
            ok = False
            break
        seg[:] = 1
    if ok:
        areas.append(int(occ.sum()))

print(f'rows {len(sub)}  no-plume {nempty}  with-footprint {len(sub)-nempty}')
if areas:
    print('footprint area px: min %d  median %d  max %d  (%.2f%% of grid median)'
          % (min(areas), int(np.median(areas)), max(areas), 100 * np.median(areas) / (H * W)))
print('ERRORS:', len(errs))
for e in errs[:10]:
    print('  -', e)
print('VALID' if not errs else 'INVALID')
