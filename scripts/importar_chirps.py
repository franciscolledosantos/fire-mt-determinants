"""Descompacta zips do ClimateSERV vindos dos Downloads e nomeia o .nc pelo ano lido no próprio arquivo."""
import sys, zipfile, os, xarray as xr, tempfile, shutil
out=sys.argv[1]
for z in sys.argv[2:]:
    with zipfile.ZipFile(z) as zf:
        for n in zf.namelist():
            tmp=tempfile.mkdtemp(); p=zf.extract(n,tmp)
            ds=xr.open_dataset(p); yrs=sorted(set(ds.time.dt.year.values)); nt=ds.sizes["time"]; ds.close()
            dst=os.path.join(out,f"chirps_MT_{yrs[0]}.nc") if len(yrs)==1 else os.path.join(out,os.path.basename(n))
            shutil.move(p,dst); shutil.rmtree(tmp); print(os.path.basename(z)[:8], "->", os.path.basename(dst), "dias:", nt)
