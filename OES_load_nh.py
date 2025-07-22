import pandas as pd
import sys
import os

target_id = str(sys.argv[1])
print(f'target ID = {target_id}')
home = os.environ['HOME']
d = f'{home}/Dropbox/share/work/astronomy/Halosat/Eridanus/analysis/OES_fuller_txt.txt'
f = pd.read_table(d,dtype='str')
obsid = list(f['Obsid'])
print(obsid)
if target_id in obsid:
    OES_nh = float(f[f.Obsid == target_id]['nh'])
else:
    print("ID check not over")
    print('Error: configuration failed', file=sys.stderr)
    sys.exit(1)
with open('fuller_nh.txt','w') as o:
    print(OES_nh,file=o)
