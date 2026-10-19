import sys, numpy as np, shutil, os
out=sys.argv[1]
for f in sys.argv[2:]:
    h=np.fromfile(f,dtype=np.int32,count=4)
    if h[0]!=0x4c55: print("ignorado",f); continue
    dst=os.path.join(out,f"mapbiomas_c10_MT005_{h[1]}.bin"); shutil.copy(f,dst); print(os.path.basename(f)[:8],"->",os.path.basename(dst))
