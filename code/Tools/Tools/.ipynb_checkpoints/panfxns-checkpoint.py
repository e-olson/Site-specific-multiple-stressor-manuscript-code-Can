"""# functions for use on PPAN system
"""
import subprocess
import os
import re
import datetime as dt
import pandas as pd
import netCDF4 as nc
from dataclasses import dataclass
from commonfxns import pidlist, pidclose, prepb19, read_torig
import socket

print('hostname:',socket.gethostname())

@dataclass
class resPaths():
    """ class for holding paths to and filenames associated with various simulations
    """
    # name associated with results group
    ident: str
    # path to pp directory on archive
    pp_path: str
    # dictionary mapping variable names and frequencies to an identifier of the file they can be found in
    # # should contain (variable, frequency) keys and path from pp directory values
    varfilemap: dict

    def dailyP(self,vfmapkey):
        return os.path.join(pp_path,varfilemap(vfmapkey))
    
historical=resPaths(ident='ESM4_historical',
                    pp_path='/archive/oar.gfdl.cmip6/ESM4/DECK/ESM4_historical_D1/gfdl.ncrc4-intel16-prod-openmp/pp/',
                    varfilemap={('sossq','daily'):'ocean_daily_gfdl/ts/daily/5yr/',
                                }
                    )

def read_file_tbounds(fpath,tbvar='time_bnds'):
    with nc.Dataset(fpath) as ff:
        torig=read_torig(ff,tbvar,'datetime')
        tunit=ff.variables[tbvar].units
        if re.match('[Dd]ays since',tunit):
            t0=torig+dt.timedelta(days=ff.variables[tbvar][:].data[0,0])
            t1=torig+dt.timedelta(days=ff.variables[tbvar][:].data[-1,-1])
        elif re.match('[Ss]econds since',tunit):
            t0=torig+dt.timedelta(seconds=ff.variables[tbvar][:].data[0,0])
            t1=torig+dt.timedelta(seconds=ff.variables[tbvar][:].data[-1,-1])
        else:
            raise Exception('units unclear: //'+tunit+'// in file '+fpath)
    return fpath, t0, t1

def cpfromTape(flist,dest='/work/Elise.Olson',maxproc=6,verb=False):
    # verb=True prints Popen stdout
    if type(flist)==str: # if single path string passed, convert to list
        flist=[flist,]
    files_onwork, files_missing = checkDestPaths(flist,dest)
    if len(files_missing)>0:
        list_DUL, list_OFL = dmCheck(files_missing)
        pid = dmgetOFL(list_OFL,wait=True)
        pids=pidlist()
        for ifile in files_missing:
            print(ifile)
            jj=0
            pids.wait(maxproc-1,verb=verb)
            pids.append(subprocess.Popen(prepb19('gcp -cd '+ifile+' '+dest+ifile), 
                      shell=True, stdout=subprocess.PIPE,  stderr=subprocess.PIPE))
        pids.wait()
        files_onwork, files_missing =checkDestPaths(flist,dest)
        if len(files_missing)>0:
            raise Exception('Some files were not copied successfully: ', files_missing)
    return files_onwork
# alias so old name should work:
cpfromArchive=cpfromTape

def clearFromWork(files_onwork,verb=False):
    for ifile in files_onwork:
        if not ( ifile.startswith('/work/Elise.Olson/archive') or ifile.startswith('/work/Elise.Olson/uda') \
              or ifile.startswith('/work/ebo/archive') or ifile.startswith('/work/ebo/uda') ):
            raise ValueError(f'file is not in a known temporary directory: {ifile}')
        if os.path.isfile(ifile):
            os.remove(ifile)
            if verb==True:
                print(f'Removed {ifile}')
    return

def checkDestPaths(flist,pfix='/work/Elise.Olson'):
    """ Check if files are already present on /work
    """
    files_missing=list()
    files_onwork=list()
    files_bad=list()
    for ff in flist:
        if os.path.exists(pfix+ff):
            files_onwork.append(pfix+ff)
        elif os.path.exists(ff):
            files_missing.append(ff)
        else:
            files_bad.append(ff)
    if len(files_bad)>0:
        raise Exception('File(s) not found:'+'\n'.join(files_bad))
    return files_onwork, files_missing   

def dmgetOFL(list_OFL,wait=True):
    """ dmget OFL files
    """
    pid=subprocess.Popen('dmget '+' '.join(list_OFL), 
                  shell=True, stdout=subprocess.PIPE,  stderr=subprocess.PIPE)
    if wait:
        pid.wait()
        for line in pid.stdout:
            print(line)
        print('dmget return code:',pid.returncode)
        pid.stdout.close()
        pid.stderr.close()
    return pid

def dmCheck(files_missing,tries=0):
    """ Check status of files on /archive or /uda
    """
    rex=re.compile(r'(?<=\()...')
    rexf=re.compile('/.*$')
    list_DUL=list()
    list_OFL=list()
    pid=subprocess.Popen('dmls -l '+' '.join(files_missing),
                shell=True,stdout=subprocess.PIPE,  stderr=subprocess.PIPE)
    pid.wait()
    for line in pid.stdout:
        stat=rex.search(line.decode('utf-8'))[0]
        if stat=='DUL':
            list_DUL.append(rexf.search(line.decode('utf-8'))[0])
        elif stat=='REG': # treat REG as DUL for purposes of this check since REG is available
            list_DUL.append(rexf.search(line.decode('utf-8'))[0])
        elif stat=='OFL':
            list_OFL.append(rexf.search(line.decode('utf-8'))[0])
        elif stat=='UNM':# treat unmigrating as offline since not yet available?
            list_OFL.append(rexf.search(line.decode('utf-8'))[0])
        elif stat=='N/A': # try waiting and rerunning; if persists, raise
            if tries<5: # up to 5 minutes
                sleep(60) # wait a minute
                list_DUL, list_OFL = dmCheck(files_missing,tries=tries+1)
            else:
                raise Exception(f'Unexpected file status:{stat} \n {line}')
        else:
            raise Exception(f'Unexpected file status:{stat} \n {line}')
    return list_DUL, list_OFL

def untar(filepath,listall=False,extractfile=''):
    """ untar file/display list of contents
    """
    if listall:
        pid=subprocess.Popen('tar -tvf '+filepath,
                shell=True,stdout=subprocess.PIPE,  stderr=subprocess.PIPE)
        pid.wait()
        for line in pid.stdout:
            print(line.decode('utf-8').strip())
    cmd='cd '+os.path.dirname(filepath)+'; tar -xvf '+' '.join([os.path.basename(filepath),extractfile])
    print(cmd)
    pid=subprocess.Popen(cmd,
           shell=True,stdout=subprocess.PIPE,  stderr=subprocess.PIPE)
    pid.wait()
    filesout=list()
    for line in pid.stdout:
        filesout.append(line.decode('utf-8').strip())
    for line in pid.stderr:
        print(line.decode('utf-8').strip())
    return filesout 

