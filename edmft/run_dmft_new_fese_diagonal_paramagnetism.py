#!/usr/bin/env python
# @Copyright 2007 Kristjan Haule
import sys, os, re, glob, shutil, socket, time
from os.path import getsize
from copy import deepcopy
import subprocess
from functools import reduce
from numpy import *
import fileinput
###
from wstruct import Struct
from scf_merge import scf, scfm
import force_stop_conditon as fstc
import utils
import indmffile
import convert_processes
import norg_fese_diagonal_paramagnetism as norg
from imp2lattc import ImpurityLatticeConnection, SimplifySiginds, iSigindDropNegative, ConvertFromLatticeToImp, Connect_ImpurityProjector
from extractInfo import FindLatticeNd_nondiag
import builtins

utime = '/usr/bin/time'  # This time function is compatible with w2k timing


def validate_fix_N_subspace_parameters(all_params):
    """Validate fixed, possibly degenerate N-subspaces in every impurity block."""
    sequence_types = (list, tuple, ndarray)
    for name, impurity_params in all_params.items():
        if not name.startswith('iparams') or not isinstance(impurity_params, dict):
            continue
        if not int(impurity_params.get('if_fix_N_subspace', [0])[0]):
            continue

        setting = impurity_params.get('fix_N_subspace')
        if not isinstance(setting, (list, tuple)) or len(setting) != 3:
            raise ValueError(
                name + "['fix_N_subspace'] must be "
                "[number_of_subspaces, subspaces, comment]"
            )

        number_of_subspaces, subspaces, _ = setting
        if (not isinstance(number_of_subspaces, (int, integer))
                or int(number_of_subspaces) <= 0):
            raise ValueError(
                name + "['fix_N_subspace'][0] must be a positive integer"
            )
        if not isinstance(subspaces, sequence_types):
            raise ValueError(
                name + "['fix_N_subspace'][1] must be a sequence of subspaces"
            )
        if int(number_of_subspaces) != len(subspaces):
            raise ValueError(
                name + "['fix_N_subspace']: declared %d subspaces, got %d"
                % (int(number_of_subspaces), len(subspaces))
            )
        if builtins.any(
                not isinstance(block, sequence_types) or len(block) == 0
                for block in subspaces):
            raise ValueError(
                name + "['fix_N_subspace'] contains an empty or invalid subspace"
            )


class Params:
    """Class to store various control paramers which control the main flow of the program
       The class is constructed such that it can handle two types of parameters:
         - starting parameters which are read only at the instantiation
         - on the fly parameters which can be refreshed (reread from the file) at any time
           by a member function refresh

       Input:
          f_sparams    -- filename of the parameters read only at the beginning
          sparams      -- a dictionary of pairs {name of the variable: default value} which are looked for in the f_sparams file
          f_params     -- filename of the parameters which can be refreshed several times by refresh member function
          params       -- a dictionary of {variable: default value} pairs which can be updated at runtime
       Output:
          None
       Variables which become members of the class are those listed in params and sparams dictionaries
    """
    def __init__(self, f_params, params, f_sparams=None, fh_info=sys.stdout):
        self.log = fh_info
        self.p ={}
        self.p.update(params) # default values from the __main__
        for var,val in self.p.items():
            print(var,val, file=self.log)
        # remember params and its data file
        self.f_params = f_params
        # values from the 'params.dat' file
        # executes the parameters at startup
        if f_sparams is not None:
            exec(compile(open(f_sparams, "rb").read(), f_sparams, 'exec'))
            sp = locals().copy()      # takes all variables currently defined here
            del sp['self']            # deletes some locals
            del sp['params']          # 
            self.p.update(sp)         # updates variables
        for var,val in self.p.items():
            print(var,val, file=self.log)
        
    def refresh(self):
        with open(self.f_params, 'rb') as config:
            settings = compile(config.read(), self.f_params, 'exec')
        sp = {}
        exec(settings, globals(), sp)
        if sp.get('solver', self.p.get('solver', 'NORG')) != 'NORG':
            raise ValueError("This paramagnetic driver requires solver='NORG'.")
        if sp.get('Ferro', self.p.get('Ferro', False)):
            raise ValueError("Ferro is incompatible with the paramagnetic driver.")
        for name, impurity_params in sp.items():
            if name.startswith('iparams') and isinstance(impurity_params, dict):
                try:
                    norg.validate_diagonal_params(impurity_params)
                    norg.green_krylov_parameters(impurity_params)
                except ValueError as exc:
                    raise ValueError("%s: %s" % (name, exc)) from exc
        # prints what was changed
        for c in sp:
            if (c not in self.p) or (sp[c] != self.p[c]):
                print('********------ parameter change ------***********', file=self.log)
                if c in self.p:
                    if isinstance(sp[c], dict):
                        for var,val in sp[c].items():
                            print('  ',c+'['+var+']->', val, file=self.log)
                    else:
                        print(c+"->", sp[c], '(',self.p[c],')', file=self.log)
                else:
                    if isinstance(sp[c], dict):
                        for var,val in sp[c].items():
                            print('  ',c+'['+var+']->', val, file=self.log)
                    else:
                        print(c+"->", sp[c], '(None)', file=self.log)
        # updates the dictonary
        self.p.update(sp)
    
    def __getitem__(self, x):
        return self.p[x]
    
    def __setitem__(self, i, value):
        self.p[i]=value
    
    #def keys(self):
    #    return list(self.p.keys())

def dmft1(fday, case, fh_info, extn, dmfe):
    inp_name = 'indmf1'
    name = 'dmft1'
    if not os.path.isfile(case+'.'+inp_name):
        print('  stop error: the required input file '+case+'.'+inp_name+' for the next step could not be found!')
        print('  stop error: the required input file '+case+'.'+inp_name+' for the next step could not be found!', file=fday)
    if not os.path.isfile(name+'.def'):
        print('  stop error: the required input file '+name+'.def for the next step could not be found!')
        print('  stop error: the required input file '+name+'.def for the next step could not be found!', file=fday)
        
    print('>%-10s' % name,  '( '+time.strftime("%H:%M:%S")+' )', file=fday)
    fday.flush()
    
    with open(':log', 'a') as fl:
        #print(time.strftime("%a %b %d %H:%M:%S %Z %Y")+'>     '+name, file=fl)
        print(time.strftime("%a %b %d %I:%M:%S %p %Z %Y")+'>     '+name, file=fl)
        
    print('Running ---- dmft1 -----', file=fh_info)
    #cmd = utime+' '+dmfe.MPI2+' '+dmfe.ROOT+'/dmft '+name+'.def >> '+name+'_info.out '
    cmd = dmfe.MPI2+' '+dmfe.ROOT+'/dmft '+name+'.def >> '+name+'_info.out '
    print('#<'+name+'>: ', cmd, file=fh_info)
    fh_info.flush()
    subprocess.call(cmd,shell=True,stdout=fh_info,stderr=sys.stderr)
    
    for fe in glob.glob('dmft1.error*'):
        if getsize(fe) !=0:
            print('ERROR in dmft1 from file:', fe, open(fe,'r').read())
            #sys.exit(1)

    
################################################
## The charge self-consistency part
################################################

# Typical Wien2k step
def AStep(fday, case, name, inp_name, WIEN, para, fh_info, opts=''):
    if not os.path.isfile(case+'.'+inp_name) and not os.path.isfile(case+'.'+inp_name+'c'):
        print('  stop error: the required input file '+case+'.'+inp_name+' for the next step could not be found!')
        print('  stop error: the required input file '+case+'.'+inp_name+' for the next step could not be found!', file=fday)
    
    tim = time.strftime("%H:%M:%S")
    print('>%-10s ( %s )' % (name+' '+opts, tim), file=fday)
    #fday.flush()
    
    cmd = WIEN+'/x'+para+' -f '+case+' '+opts+' '+name
    print(('#<'+name+'>: '), cmd, file=fh_info)
    fh_info.flush()
    
    info=subprocess.call(cmd,shell=True,stdout=fday)
    
    for fe in glob.glob(name+'.error*'):
        if getsize(fe) !=0:
            print('ERROR in '+fe+' from file:', open(fe,'r').readlines())

def lapw0(fday, case, WIEN, para, fh_info):
    para=''
    if os.path.isfile(case+'.in0_grr'):
        AStep(fday, case, 'lapw0', 'in0_grr', WIEN, para, fh_info, '-grr')
    AStep(fday, case, 'lapw0', 'in0', WIEN, para, fh_info)

        
def lapw1(fday, case, WIEN, para, dftKS, dmfe, complex_suffix, fh_info):
    pcmplx = '-c' if complex_suffix else ''
    if dftKS:
        name='lapw1'
        sname = 'lapw1'+complex_suffix
        print('>%-10s' % sname, '( '+time.strftime("%I:%M:%S %p")+' )', file=fday)
        with open(':log', 'a') as fl:
            print(time.strftime("%a %b %d %I:%M:%S %p %Z %Y")+'>     '+sname, file=fl)
        cmd = dmfe.ROOT+'/x_dmft.py'+para+' '+pcmplx+' '+name
        print(('#<'+sname+'>: '), cmd, file=fh_info)
        fh_info.flush()
        info=subprocess.call(cmd,shell=True,stdout=fh_info)
    else:
        AStep(fday, case, 'lapw1', 'in1', WIEN, para, fh_info, pcmplx)
        
def lcore(fday, case, WIEN, fh_info):
    AStep(fday, case, 'lcore', 'inc', WIEN, '', fh_info)

def mixer(fday, case, WIEN, fh_info):
    AStep(fday, case, 'mixer', 'inm', WIEN, '', fh_info)

def dmft2(fday, case, fh_info, dmfe):
    inp_name = 'indmf2'
    name = 'dmft2'
    if not os.path.isfile(case+'.'+inp_name):
        print('  stop error: the required input file '+case+'.'+inp_name+' for the next step could not be found!')
        print('  stop error: the required input file '+case+'.'+inp_name+' for the next step could not be found!', file=fday)
    if not os.path.isfile(name+'.def'):
        print('  stop error: the required input file '+name+'.def for the next step could not be found!')
        print('  stop error: the required input file '+name+'.def for the next step could not be found!', file=fday)
        
    print('>%-10s' % name, '( '+time.strftime("%I:%M:%S %p")+' )', file=fday)
    fday.flush()
    
    with open(':log', 'a') as fl:
        print(time.strftime("%a %b %d %I:%M:%S %p %Z %Y")+'>     '+name, file=fl)
    
    #cmd = utime+' '+dmfe.MPI2+' '+dmfe.ROOT+'/dmft2 '+name+'.def >> dmft2_info.out '
    cmd = dmfe.MPI2+' '+dmfe.ROOT+'/dmft2 '+name+'.def >> dmft2_info.out '
    print(('#<'+name+'>: '), cmd, file=fh_info)
    fh_info.flush()
    
    info=subprocess.call(cmd,shell=True,stdout=fh_info,stderr=sys.stderr)
    
    for fe in glob.glob(name+'.error*'):
        if getsize(fe) !=0:
            print('ERROR in '+fe+' from file:', open(fe,'r').readlines())
            #sys.exit(1)
    
    dat = open(case+'.scf2').readlines()
    dEF=0
    for line in dat:
        if line[:5]==':DEF ':
            dEF = float(line.split()[1])
            break
    
    return dEF
    
def scf1(fday, case, WIEN):
    for i in ['clmsum','vsp','vns','vrespsum']:
        name = case+'.'+i
        if os.path.exists(name) and os.path.getsize(name)>0:
            shutil.copy2(name, name+'_old')		#save last cycle
            
        
def Diff(fday, case, dEF, diff_nimp, ddiff_nimp):
    #
    with open(case+'.scf', 'r') as fs:
        dat = fs.readlines()
    
    DEne=[]
    DChr=[]
    for line in dat:
        if re.match(r':ENE', line):
            DEne.append(float(line.split()[-1]))
            #print line.split()[-1]
        if re.match(r':DIS', line):
            DChr.append(float(line.split()[-1]))
            #print line.split()[-1]
    if len(DChr)>1:
        drho = DChr[-1]
    else:
        drho = 1.0
    if len(DEne)>1:
        dene = abs(DEne[-1]-DEne[-2])
    else:
        dene = 1.0
    if dene==0: dene = 1.0    
    print(':ENERGY convergence: '+str(dene), file=fday)
    print(':CHARGE convergence: '+str(drho), file=fday)
    print(':EF     convergence: '+str(dEF),  file=fday)
    
    print(':ENERGY convergence: '+str(dene))
    print(':CHARGE convergence: '+str(drho))
    print(':EF     convergence: '+str(dEF))
    print(':NIMP   convergence: '+str(ddiff_nimp))
    print(':NIMP   difference : '+str(diff_nimp))
    return (drho, dene)


def SolveImpurity(EF, asolver, iat, extn, p, UpdateAtom, nl_imp, fh_info, fh_pinfo, fday):
    #print('Running ----- impurity solver -----')
    print('UpdateAtom=', UpdateAtom, file=fh_info)
    print('Running ----- impurity solver -----', file=fh_info)
    fh_info.flush()
    
    tim = time.strftime("%I:%M:%S %p")
    print('>%-10s ( %s )' % ('impurity', tim), file=fday)
    fday.flush()
    with open(':log', 'a') as fl:
        print(time.strftime("%a %b %d %I:%M:%S %p %Z %Y")+'>     '+'impurity', file=fl)
    
    iprms = 'iparams'+str(iat)
    ipars = deepcopy(p[iprms])
    
    asolver._exe(ipars, p['DCs'], extn, UpdateAtom, p['wbroad'], p['kbroad'])
    fh_info.flush()
    
    print('Taking care of high frequency', file=fh_info)
    ntot = asolver.HighFrequency(ipars, p['DCs'], nl_imp, extn, p['wbroad'], p['kbroad'])
    
    with open('imp.'+str(iat)+'/sig.out','r') as fhi:
        print('NORG.HighFrequency(s_oo) :\n', next(fhi), next(fhi), file=fh_info)
    with open('imp.'+str(iat)+'/Eorb.dat', 'r') as fhi:
        print('NORG.HighFrequency(Eorb) :\n', fhi.read(), file=fh_info)
    
    # Reading impurity levels for printing
    (Eimp,Olap,Edc) = loadtxt('imp.'+str(iat)+'/Eimp.inp',ndmin=2)   
    #fEimp = open('imp.'+str(iat)+'/Eimp.inp', 'r')
    #(Eimp,Olap,Edc) = [array([float(x) for x in line.split('#')[0].split()]) for line in fEimp.readlines()]  # throw out comments
    print('Eimp=', Eimp-Edc, file=fh_info)
    print('Edc=', Edc, file=fh_info)
    print('ncorr=', ntot, file=fh_info)
    #icycle, itt = extn.split('.')
    #print >> fh_pinfo, '%3s.%3s %12.6f %12.6f %12.6f %12.6f %12.6f' % (icycle, itt, EF, Eimp[0], Eimp[-1], Edc[0], ntot)
    #fh_pinfo.flush()
    return ntot

def SSplit(ROOT, matsubara, iparams, ntail, fh_info):
    cmd = ROOT+'/ssplit.py'
    if matsubara : cmd += ' -n '+str(iparams['nom'][0])+' -t '+str(ntail)
    print('#<ssplit>: ', cmd, file=fh_info)
    info=subprocess.call(cmd,shell=True,stdout=fh_info,stderr=fh_info)
    
def SGather(ROOT, fh_info, p):
    cmd = ROOT+'/sgather.py' 
    if p['mix_sigma']<1.0 : 
        cmd += ' -m '+str(p['mix_sigma'])
    print('#<sgather>: ', cmd, file=fh_info)
    info=subprocess.call(cmd,shell=True,stdout=fh_info,stderr=fh_info)

def SJoin(ROOT, fh_info, p):    
    cmd = ROOT+'/sjoin.py -m '+str(p['mix_delta'])
    if p['rCF'] is not None: cmd += ' -c "'+ p['rCF']+'"'
    print('#<sjoin>: ', cmd, file=fh_info)
    info=subprocess.call(cmd,shell=True,stdout=fh_info,stderr=fh_info)

def Prepare_dmft1(dmfe, para, fh_info):
    print('--------- Preparing dmft1 calculation ---------', file=fh_info)
    cmd = dmfe.ROOT+'/x_dmft.py -d' + para + ' dmft1 >> dmft1_info.out 2>&1'
    print('#<prep-dmft1>: ', cmd, file=fh_info)
    info=subprocess.call(cmd,shell=True,stdout=fh_info,stderr=fh_info)

def Prepare_dmft2(dmfe, p, para, fh_info, recomputeEF, mode='c'):
    print('--------- Preparing dmft2 calculation ---------', file=fh_info)
    cmd = dmfe.ROOT+'/x_dmft.py -d '+'--mode '+mode+' -m '+str(recomputeEF)+' -x '+str(p['mixEF'])+' -w '+str(p['WL'])+para+' dmft2 >> dmft2_info.out 2>&1'
    print('#<prep-dmft2>: ', cmd, file=fh_info)
    info=subprocess.call(cmd,shell=True,stdout=fh_info,stderr=fh_info)

def CreateEorb(case):
    """ Reads case.scf2 file (prepared by dmft2) and extracts 
    density matrix, (density on each atom) which starts with :NCOR.
    Also gives the first component of double-counting
    """ 
    with open('sig.inp','r') as fsg:
        exec(next(fsg)[1:].lstrip(), globals())  # s_oo from sig.inp                                                                                                                   $
        exec(next(fsg)[1:].lstrip(), globals())  # Edc  from sig.inp                                                                                                                   $
    with open(case+'.scf2','r') as fsc:
        dat = fsc.readlines()
    nd=[]
    for l in range(len(dat)):
        if dat[l][:5]==':NCOR':
            dd = dat[l].split()
            nd.append( float(dd[1]) )
    return (nd,Edc[0])

def FindEtot(case):
    def FindLine(dat, strng, item):
        Val = 0.0
        for line in dat:
            if re.match(strng, line) is not None:
                Val = float(line.split()[item])
                break
        return Val

    with open(case+'.scfm','r') as fs:
        dat = fs.readlines()[::-1]  # it sorts from the end back
    ETOT = FindLine(dat, ':ENE', 8) # last :ENE

    if os.path.isfile('Eorb_imp.dat'):
        with open('Eorb_imp.dat','r') as fs:
            dat = fs.readlines()[::-1]
        XEORB = FindLine(dat, ':XEORB',1)
        EORB  = FindLine(dat, ':EORB',1)
        ZEORB = FindLine(dat, ':ZEORB',1)
    else:
        XEORB = EORB = ZEORB = 0.
    
    with open(case+'.scf2','r') as fs:
        dat = fs.readlines()[::-1]
    SUM  = FindLine(dat, ':SUM ', 6)
    XSUM = FindLine(dat, ':XSUM', 6)
    YSUM = FindLine(dat, ':YSUM', 6)
    ZSUM = FindLine(dat, ':ZSUM', 6)
    return (ETOT,SUM,XSUM,YSUM,ZSUM,EORB,XEORB,ZEORB)

def read_norg_initial_occupations(filename, solver, cix_map):
    """Select the specified lattice sites and return paired PM occupations."""
    if (not isinstance(cix_map, (list, tuple)) or len(cix_map) != 3
            or list(cix_map[:2]) != [1, 1]
            or not isinstance(cix_map[2], (list, tuple)) or not cix_map[2]):
        raise ValueError("The paramagnetic cix_map must be [1, 1, [cix, ...]].")
    selected = cix_map[2]
    density = {}
    with open(filename) as stream:
        for line in stream:
            fields = line.split()
            if not fields or fields[0] != ':NCOR':
                continue
            cix, size = int(fields[2]), int(fields[3])
            rows = [next(stream).split() for _ in range(size)]
            if cix not in selected or cix in density:
                continue
            diag_values = array([float(row[2*i]) for i, row in enumerate(rows)])
            if size == len(solver.Sigind_shifted):
                diag_values = diag_values[solver.active_rows]
            elif size != len(solver.active_rows):
                raise ValueError("Occupation dimension for cix=%s does not match Sigind" % cix)
            density[cix] = diag_values
    missing = [cix for cix in selected if cix not in density]
    if missing:
        raise ValueError("Missing cix %s in %s" % (missing, filename))
    per_spin = mean([density[cix] for cix in selected], axis=0) / 2.0
    return repeat(per_spin, 2).tolist()


if __name__ == '__main__':
    # -------------- Parameters which are set through input files sparams.dat and params.dat -----------
    # Default values of some parameters
    params = {'runIMP'        : True,      # Run impurity solver
              'runGF'         : True,      # Run dmft1 for G and Delta
              'UpdateAtom'    : 1,         # recompute cix-file for the impurity solver
              'DCs'           : 'nominal',    # the double counting scheme, which fixes Edc through n0
              'max_dmft_iterations': 1,    # number of iteration of the dmft-loop only
              'max_lda_iterations' : 1,    # number of iteration of the LDA-loop only
              'max_lda_iterations_optimize' : 1000,    # number of iteration of the LDA-loop when structure should be optimized
              'finish'        : 50,        # number of iterations of the charge loop
              'da'            : 0.0,       # experimental
              'wbroad'        : 0.0,       # broadening of sigma on the imaginary axis
              'kbroad'        : 0.0,       # broadening of sigma on the imaginary axis
              'solver'        : 'NORG',    # diagonal paramagnetic solver
              'ntail'         : 200,       # on imaginary axis, number of points in the tail of the logarithmic mesh
              'mix_delta'     : 1.0,       # whether to mix delta, or not
              'mix_sigma'     : 1.0,       # whether to mix sigma, or not
              'riter'         : 100,        # How often to restart broyden for charge mixing
              'sleeptime'     : 2,         # If broyden file are present at the submit time, user has some time to kill the job
              'cc'            : 1e-5,      # the charge density precision to stop the LDA+DMFT run
              'ec'            : 1e-5,      # the energy precision to stop the LDA+DMFT run
              'nc'            : 1e-10,     # the impurity difference to stop LDA+DMFT run
              'rCF'           : None,      # Reduction of the crystal field splitting, if necessary
              'recomputeEF'   : 1,         # Recompute EF in dmft2 step. If recomputeEF=2, it tries to find an insulating gap.
              'mixEF'         : 1.0,       # Chemical potential can be mixed with mixEF<1.0
              'saver'         : 0.0,       # Average over the last few DMFT self-energies is used for charge. Should be between [0,1]
              'WL'            : 1.0,       # If recomputeEF=2, we look for the pole in the interval [-WL,WL]
              'dftKS'         : True,      # Use internal capabilities to solve the Kohn-Sham problem
              'remove_broyd'  : 1000,      # How many dft steps before broyden files are removed
              'force_Ene_diff': 0.001,     # When checking if force is converged, we require Energy to be converged as well up to this difference
              'force_Rho_diff': 0.01,      # When checking if force is converged, we require Charge to be converged as well up to this difference
              }

    Ry2eV = utils.Ry_in_eV

    if len(sys.argv)>1 and sys.argv[1] in ['-h', '--help']:
        help="""
        The script runs LDA+DMFT. It is a wrapper, which calls other scripts and executables.
        It executes Wien2K in parallel using dftKS program (mpi parallelization over k-points)
        or it can use Wien2k parallelization. 
        The DMFT part will run in parallel even if DFT part is sequential.
        Usuall execution goes through the following steps in a loop:
        
          x_lapw    lapw0        -- computes LDA potential on current LDA+DMFT charge
          x_dmft.py lapw1        -- solves LDA eigenvalue problem
          x_dmft.py  dmft1       -- computes local green's function and hybridization
          run impurity   -- runs impurity problem
          x_dmft.py dmft2        -- computes LDA+DMFT valence charge, and the chemical potential
          x_lapw lcore        -- computes LDA core charge
          x_lape mixer        -- mixes total charge with the previous charge

        The most common parameters, which should be given through 'params.dat' file, include:

        name       possible values   default     help
        --------------------------------------------------------------------------------------
        dftKS          [True|False]  True     # use internal LAPW1
        solver         [NORG]        NORG     # diagonal paramagnetic impurity solver
        max_iterations [int]         1        # number of iteration of the dmft-loop only
        finish         [int]         100      # number of iterations of the charge+dmft loop
        cc             [float]       1e-5,    # the charge density precision to stop the LDA+DMFT run
        ec             [float]       1e-5,    # the energy precision to stop the LDA+DMFT run
        broyd          [True| False] True     # Are we using broyden for charge mixing
        riter          [int]         99       # How often to restart broyden for charge mixing
        sleeptime      [int]         2        # If broyden file are present at the submit time, user
                                                # has some time to kill the job
        DCs            [exacty | exactd | exact | nominal | fixn | FLL | AMF | exact_l ]
                       exacty : dielectric+yukawa screening, such that we get predescribed U & J. Real space form is e^{-lambda*r}/(eps*r)
                       exactd : dielectric screening only, i.e., U_{1234}=<psi_1|psi_2|U|psi_3|psi_4>/eps. U is always make equal to input U, but J is not, and is typically smaller.
                       exact  : dielectric+yukawa screening, first computing lambda,epsilon from U&J, and then renormalizing
                                lambda by fractional occupancy of the shell (very little yukawa screening for late TMO and mostly yukawa for early TMOs).
                                dielectric screening is later increased to compensate for descreased lambda, to give input U. But J is probably smalller 
                                than required by input.
        runIMP         [True| False] True     # Run impurity solver or skip this step
        runGF          [True| False] True     # Run dmft1 for G and Delta, or skip this step
        ntail          [int]         30       # sigma.inp on imaginary axis, number of points in the
                                                # tail of the logarithmic mesh
        mix_delta      [float]       1.0      # linear mixing parameter for Delta.
        UpdateAtom     [int]         0        # How often to recompute cix-file inside the
                                              # impurity solver
        wbroad         [float]       0.0      # constant broadening of sigma on the imaginary axis
        kbroad         [float]       0.0,     # linear broadening of sigma on the imaginary axis
        recomputeEF    [0|1|2]       1        # If we want to recompute the chemical potential in dmft2 step. 2->for Mott insulator
        com            [0| 1]        0        # when computing chemical potential in [dmft0|mu] step
                                                # on imaginary axis, we can subtract Sigma_oo (com=0)
                                                # or Sigma(0) (com=1)
        mixEF          0.0-1.0                # mixing of the chemical potential
        da             [float]       0.0      # experimental - connected with DC scheme
       
        """
        print(help)
        sys.exit(0)
    
    fh_info = open('dmft_info.out','w')
    fh_pinfo = open('info.iterate', 'a')
    
    p = Params('params.dat', params, f_sparams=None, fh_info=fh_info)
    p.refresh()
    validate_fix_N_subspace_parameters(p.p)
    complex_suffix = ''
    dmfe = utils.DmftEnvironment()  # DMFT paths
    w2k = utils.W2kEnvironment()    # W2k filenames and paths
    case = w2k.case
    if (os.path.isfile(case + '.indmfldn') or os.path.isfile(case + '.scfdmft1dn')
            or builtins.any(os.path.isfile(case + suffix) and os.path.getsize(case + suffix) > 0
                            for suffix in ('.incup', '.incdn'))):
        raise ValueError("Diagonal paramagnetic NORG requires one spin-degenerate lattice input.")
    if os.path.isfile(case + '.inso') and os.path.getsize(case + '.inso') > 0:
        raise ValueError("Diagonal paramagnetic NORG requires a spatial d basis without spin-orbit mixing.")
    para = ''
    if p['dftKS'] and dmfe.MPI2:
        para = ' -p'  # Parallel run with internal lapw1. We could use either '-p' to have many vector files, or just '', to have single vector file. Not clear what is faster...
    if os.path.isfile('.machines') and os.path.getsize('.machines')>0 : 
        para = ' -p'              # Using WIEN2k parallelization.
        p['dftKS'] = False        # Switching off internal lapw1
    
    # we want to clean certain files that might interfere with this run
    to_clean = ['*.scf','Edc.dat','dmft[1|2].error*','*.[dlt|gc1|cdos].*','*.outputdmf[1|2].*','dmft[1|2]_info.out', '*.broyd*']
    toclean = []
    for pat in to_clean:
        toclean.extend( glob.glob(pat) )
    for f in toclean: os.remove(f)
    
    # Info files
    fday = open(w2k.case+'.dayfile', 'w')
    print('Calculating '+w2k.case+' in '+os.getcwd()+'\n'+'on '+socket.gethostname()+' with PID '+str(os.getpid()), file=fday)
    print('\n\n', file=fday)
    
    
    # Reading 'case.indmfl' file
    inl = indmffile.Indmfl(w2k.case) # case.indmfl file
    inl.read()                       # case.indmfl read

    print('inl.emin={:s}, inl.emax={:s} inl.hybr_emin={:f} inl.hybr_emax={:f}'.format(str(inl.emin),str(inl.emax),inl.hybr_emin,inl.hybr_emax), file=fh_info)
    if not inl.matsubara:
        raise ValueError("Diagonal paramagnetic NORG requires Matsubara input.")
    
    ## produce projectorw.dat if it does not yet exist.
    if inl.projector>=4 and not os.path.exists('projectorw.dat'):
        print('WARNING: projectorw.dat does not exist, hence trying to computed it. Make sure case.vsp is available', file=fh_info)
        fh_info.flush()
        if not os.path.isfile(w2k.case+'.vsp'):
            print('ERROR: projectorw.dat does not exist nor case.vsp, hence I can not compute projector. Run "init_proj.py" first.', file=fh_info)
            print('ERROR: projectorw.dat does not exist nor case.vsp, hence I can not compute projector. Run "init_proj.py" first.')
            sys.exit(1)
        inl.CmpProjector(log=fh_info)
    ## if the band range is not yet specified, do it now
    if inl.emin==inl.emax==0: # we did not yet determine band ranges
        inl.CmpBandRange(log=fh_info)
        inl.write(w2k.case+'.indmfl')

    # Reads structure file
    struct = Struct(inAngs=False)
    struct.ReadStruct(w2k.case+'.struct', fh_info)
    
    if os.path.isfile(w2k.case+".in1c") and os.path.getsize(w2k.case+".in1c")>0 :
        print('Found '+w2k.case+'.in1c file, hence assuming non-centrosymmetric structure. Switching -c switch!', file=fh_info)
        complex_suffix = 'c'
        
    # corelated indexes
    cixs = list(inl.siginds.keys())        # all columns for cix
    
    # Nuclear charges of all correlated blocks
    Znuc={} 
    for icix in cixs:
        atm = inl.cix[icix][0][0]                # correlated atoms
        Znuc[icix] = int(round(struct.aZnuc[atm-1]))  # nuclear charge of each correlated block
        
    print('Znucs=', Znuc, file=fh_info)

    # Independent impurity problems
    iSigind = indmffile.ParsIndmfi(w2k.case)
    print('iSiginds=', iSigind, file=fh_info)
    cols = SimplifySiginds(inl.siginds)
    icols = SimplifySiginds(iSigind)
    imp2latt = ImpurityLatticeConnection( cols, icols, fh_info)
    print('Impurity-lattice connection: imp2latt=', imp2latt, file=fh_info)
    
    fh_pinfo.write( '%3s %3s.%4s %12s %12s ' % ('#', '#','#','mu','Vdc') )
    fh_pinfo.write( '%15s %15s %15s ' % ('Etot', 'Ftot+T*Simp', 'Ftot+T*Simp') )
    for iat in list(iSigind.keys()):   # Over all inequivalent impurity problems
        fh_pinfo.write( '%12s %12s %12s %12s ' % ('n_latt', 'n_imp','Eimp[0]','Eimp[-1]') )
    fh_pinfo.write('\n')
    fh_pinfo.flush()


    if p['DCs'][:5]=='exact':
        Nmax=[]
        for imp in list(iSigind.keys()):
            icx = imp2latt[imp][0]
            iatom, l, qsplit = inl.cix[icx][0]  # Doesn't take into account cluster problems correctly
            Nmax.append(2*(2*l+1))

        # Finds connection between impurity and projector
        impurity_projector = Connect_ImpurityProjector(imp2latt,inl,struct,fh_info)
        print('connection -- impurity_projector=', impurity_projector, file=fh_info)
        # Finds the screening length and corresponding J's given imput value of Coulomb U for al impurity problems
        UJ_icase={}
        for imp,icase in enumerate(impurity_projector):
            iprms = 'iparams'+str(imp)
            U=p[iprms]['U'][0]            # we always need U to determine screening length
            JH=0
            if 'J' in p[iprms]:    # if Hunds J is awailable, we take it into account
                JH = p[iprms]['J'][0]
            lmbda=1e-6
            if 'DC_lmbda' in p[iprms]: # if user wants to use finite lambda with exactd DC, it should set iparamsx['DC_lmbda']
                lmbda = p[iprms]['DC_lmbda'][0]
            nfraction=1.0
            if 'nf0' in p[iprms]:
                n0 = p[iprms]['nf0'][0]
                nfraction = (Nmax[imp]-n0)/(Nmax[imp]+0.0) # fractional occupancy
            UJ_icase[icase]=[U,JH,lmbda,nfraction]  # Saves U into U_icase, which is needed below to compute lambda
        
        print('UJ_icase=', UJ_icase, file=fh_info)
        import RCoulombU
        if len(p['DCs'])>5 and p['DCs'][:6]=='exactd':      # dielectric screening only
            print('Have exactd!', file=fh_info)
            UlamJ = RCoulombU.GetDielectricFunctions(UJ_icase, log=fh_info)  # Computes dielectric function epsilon and sets yukawa lambda=1e-6
        elif len(p['DCs'])>5 and p['DCs'][:6]=='exacty':    # yukawa + dielectric screening, should be best
            print('Have exacty!', file=fh_info)
            for icase in list(UJ_icase.keys()):
                UJ_icase[icase][3]=1.0 # all nfractions equal to 1.0, hence yukawa
            #print('UJ_icase=', UJ_icase)
            UlamJ = RCoulombU.GetLambdas(UJ_icase, log=fh_info) # Computes yukawa lambda and dielectric function so that they match user defined U & J.
        else :  # mixture between yukawa and dielectric. First we compute lambda&epsilon from input U&J. In the second step we reduce
                # lambda (Yukawa screening) by partial occupancy so that d^9 or d^10 system has very little or no yukawa screening, while d^0 d^1 has
                # yukawa & epsilon corresponding to input U&J. Then we increase dielectric screening to compensate for smaller Yukawa lambda.
                # This is done because in cuprates, for example, exactd seems to be the best, while in early TMO's yukawa is fine.
            print('Have exact!', file=fh_info)
            #print('UJ_icase=', UJ_icase)
            UlamJ = RCoulombU.GetLambdas(UJ_icase, log=fh_info)
            
        for imp,icase in enumerate(impurity_projector): # Saves into impurity
            iprms = 'iparams'+str(imp)
            p[iprms]['icase'] = [icase, '#'] # Connection between impurity and projector in projector.dat'
            p[iprms]['UlamJ'] = [UlamJ[icase], '#']
        
        print('UlamJ = {iatom : (U,lambda,epsilon,[J2,J4,...]), ....}', file=fh_info)
        print('UlamJ=', UlamJ, file=fh_info)
    

    solver = {}
    for ic, isigind in iSigind.items():
        if not imp2latt[ic]:  # Open-core atom
            continue
        icx = imp2latt[ic][0]
        for cix in imp2latt[ic]:
            orbital_l = inl.cix[cix][0][1]
            if orbital_l != 2 or len(inl.siginds[cix]) > 2 * orbital_l + 1:
                raise ValueError("Diagonal paramagnetic NORG requires spatial d orbitals.")
        if len(isigind) > 5:
            raise ValueError("Diagonal paramagnetic NORG requires spatial d orbitals.")
        # Check static off-diagonal labels before dropping the inactive orbitals.
        for index_matrix in [isigind] + [inl.siginds[cix] for cix in imp2latt[ic]]:
            if (index_matrix != diag(diag(index_matrix))).any():
                raise ValueError("Diagonal NORG cannot accept off-diagonal lattice or impurity Sigind.")
        isigind_ = iSigindDropNegative(isigind, icols[ic], fh_info)
        solver[ic] = norg.IMP_NORG(Znuc[icx], 'imp.' + str(ic), p['iparams' + str(ic)],
                                   isigind_, inl.cftrans[icx])

    # We decided not to copy executables anymore. Too much temp files
    #shutil.copy2(dmfe.ROOT+'/dmft', '.')  # For parallel execution, executable has to be on the current directory
    #shutil.copy2(dmfe.ROOT+'/dmft2', '.') # For parallel execution, executable has to be on the current directory


    if p['recomputeEF']==0:
        # W2k now changed mixer, so that it does not normalize charge. If we have recomputeEF=0, it will result in error. Hence we need to change case.inm
        fname = w2k.case+'.inm'
        if os.path.exists(fname):
            with open(fname, "r") as fi:
                content = fi.read()
            m = re.match(r"^(\w+\s+\d+(\.\d+)?\s+)YES", content, flags=re.MULTILINE)
            if m is not None:
                content = re.sub(r"^(\w+\s+\d+(\.\d+)?\s+)YES", r"\1NO", content, flags=re.MULTILINE)
                print('since recomputeEF=0 we need to allow unrenormalized charge in mixer. Changing case.inm to', file=fh_info)
                print(content, file=fh_info)
                with open(fname, "w") as fi:
                    fi.write(content)
        if os.path.exists(w2k.case+'.scf2') and not os.path.exists('EF.dat'):
            ef = None
            with open(w2k.case+'.scf2', 'r') as fi:
                lines = fi.readlines()
                for line in lines[::-1]:
                    if line[:4]==':FER':
                        ef = float(line.split()[-1])
                        break
            if ef is not None:
                with open('EF.dat', 'w') as fo:
                    print(ef*Ry2eV, file=fo)

    Prepare_dmft1(dmfe, para, fh_info)
    Prepare_dmft2(dmfe, p, para, fh_info, p['recomputeEF'])
    
    shutil.copy2('sig.inp', 'sig.inp.0')
    
    if p['finish']>1: # Charge self-consistent
        
        if os.path.isfile(w2k.case+'.clmsum_old'): shutil.copy2(w2k.case+'.clmsum_old', w2k.case+'.clmsum')
        if not os.path.isfile(w2k.case+'.clmsum'):
            print('no '+w2k.case+'.clmsum(_old) file found, which is necessary for lapw0 !')
            print('no '+w2k.case+'.clmsum(_old) file found, which is necessary for lapw0 !', file=fday)
            sys.exit(1)
        
        if os.path.isfile(w2k.case+'.broyd1'):
            print(w2k.case+'.broyd* files present! I will remove previous broyd files.') 
            # print 'You have '+str(p['sleeptime'])+' seconds to kill this job ( ^C   or   kill '+str(os.getpid())+' )'
            # print 'or the script will rm *.broyd* and continue (use -NI to avoid automatic rm)'
            # time.sleep(p['sleeptime'])
            cmd = 'rm -f *.broyd*'
            subprocess.call(cmd,shell=True,stdout=fh_info,stderr=fh_info)
            print(w2k.case+'.broyd* files removed!', file=fday)
        
        print('\n   start'+(' '*8)+ time.asctime()+' with lapw0 ('+str(1)+'/'+str(p['riter'])+' to go)', file=fday)
        
        # Starting with LAPW
        print('\n   cycle %s \t%s %s/%s to go\n' % (0,time.asctime(),p['finish'],(p['finish'])%p['riter']), file=fday)
        lapw0(fday,w2k.case,w2k.WIENROOT,para, fh_info); fday.flush()
        lapw1(fday,w2k.case,w2k.WIENROOT,para,p['dftKS'],dmfe,complex_suffix,fh_info); fday.flush()
        fday.flush()
        
    if para and (not p['dftKS']): convert_processes.convert('', fh_info)
    
    #####################
    # Major charge loop #
    #####################
    icycle = nimp = 0
    istep0=0
    nl_imp = None
    diff_nimp,ddiff_nimp,pdiff_nimp=1,1,0
    (RelaxingStructure, mix_method) = fstc.AreWeRelaxingStructure2(w2k.case)
    print('Relaxing Structure=', RelaxingStructure, file=fh_info)
    
    while icycle < p['finish']:

        if (icycle>0): # Does not exists yet. Will find it after dmft1
            imp_sigind_map = {cix: solver[iat].Sigind_shifted
                              for iat in list(iSigind.keys()) for cix in imp2latt[iat]}
            cix_maps = {iat: p['iparams'+str(iat)].get('cix_map', [None])[0]
                        for iat in list(iSigind.keys())}
            (nl_imp,nd_imp) = FindLatticeNd_nondiag(w2k.case, '', inl.siginds, imp2latt, fh_info, imp_sigind_map, scf_name='scf2', cix_maps=cix_maps)
        
        print('nl_imp=', nl_imp, file=fh_info)

        #####################
        # DMFT loop         #
        #####################
        sigmas=[]
        itt = 0
        n_imp = zeros(len(list(iSigind.keys())))
        
        if 'GoodGuess' in p.p and p['GoodGuess']==True:
            if icycle == 0 : 
                p['max_dmft_iterations'] = 0 # do not run impurity at the beginning, but rather converge DMFT charge on existing self-energy
                itt = 1
            if icycle == 1 : 
                p['max_dmft_iterations'] = 2 # at the next iteration, rerun impurity twice to have good histogram from the second run
                itt = 0
            if icycle > 1  : p['max_dmft_iterations'] = 1 # finally, continue as normally
            print('We have GoodGuess and max_dmft_iterations=', p['max_dmft_iterations'], 'itt=', itt, file=fh_info)
        
        while itt < p['max_dmft_iterations']:
            
            extn = str(icycle+1)+'.'+str(itt+1)  # Extension for the output files
            p.refresh()                          # updating parameters in case user changes them
            
            # Recomputing the chemical potential, but only if more than one DMFT step. Otherwise EF was recomputed in charge-SC step.
            if itt>1 and p['recomputeEF']:
                Prepare_dmft2(dmfe, p, para, fh_info, 1, 'm')
                SSplit(dmfe.ROOT, inl.matsubara, p['iparams0'], p['ntail'], fh_info)
                dEF =  dmft2(fday, w2k.case, fh_info, dmfe); fday.flush()

            # Computing Green's function
            if p['runGF']:
                SSplit(dmfe.ROOT, inl.matsubara, p['iparams0'], p['ntail'], fh_info)
                dmft1(fday, w2k.case, fh_info, extn, dmfe)
                imp_sigind_map = {cix: solver[iat].Sigind_shifted
                                  for iat in list(iSigind.keys()) for cix in imp2latt[iat]}
                cix_maps = {iat: p['iparams'+str(iat)].get('cix_map', [None])[0]
                            for iat in list(iSigind.keys())}
                (nl_imp,nd_imp) = FindLatticeNd_nondiag(w2k.case, '', inl.siginds, imp2latt, fh_info, imp_sigind_map, scf_name='scfdmft1',cmplx=True,cix_maps=cix_maps)
            
            SJoin(dmfe.ROOT, fh_info, p)

            if os.path.isfile('EF.dat'):
                EF = float(open('EF.dat').read())
            else:
                open(w2k.case+'.indmf1')
                EF =float(open(w2k.case+'.indmf1').readlines()[1].split()[1])
            
            # ------------ Impurity solver -------------------
            if p['runIMP']:
                nimp += 1
                UpdateAtom = (p['UpdateAtom']!=0 and mod(nimp,p['UpdateAtom'])==0)

                occupation_file = w2k.case + '.scfdmft1'
                if os.path.isfile(occupation_file):
                    for iat in solver:
                        ipars = p['iparams' + str(iat)]
                        cix_map = ipars.get('cix_map', [None])[0]
                        n_init = read_norg_initial_occupations(occupation_file, solver[iat], cix_map)
                        ipars['n_init'] = [n_init, '# initial occupancies per spin-orbital']
                        print('Updated impurity', iat, 'n_init:', n_init, file=fh_info)

                for iat in solver:   # Over all inequivalent impurity problems
                    print('DEBUG iparams n_init present:', 'n_init' in p['iparams'+str(iat)], file=fh_info); fh_info.flush()
                    ni=SolveImpurity(EF, solver[iat], iat, extn, p, UpdateAtom, nl_imp[iat], fh_info, fh_pinfo, fday)
                    n_imp[iat] += ni
                    print('STOP  IMPURITY_'+str(iat), file=sys.stderr)
                SGather(dmfe.ROOT, fh_info, p)

                
                IEorb = 0.0
                XEorb = 0.0
                ZEorb = 0.0
                dat=[]
                for iat in solver:   # Over all inequivalent impurity problems
                    if os.path.exists(solver[iat].dir+'/Eorb.dat'):
                        dat += ['--- impurity '+str(iat)+' with degeneracy '+str(len(imp2latt[iat]))+' ---\n']
                        datw = open(solver[iat].dir+'/Eorb.dat','r').readlines()
                        for line in datw:
                            if line[:6]==':IEORB':
                                IEorb += float(line[6:]) * len(imp2latt[iat]) # Takes into account the degeneracy of the impurity
                            elif line[:6]==':XEORB':
                                XEorb += float(line[6:]) * len(imp2latt[iat]) # Takes into account the degeneracy of the impurity  
                            elif line[:6]==':ZEORB':
                                ZEorb += float(line[6:]) * len(imp2latt[iat]) # Takes into account the degeneracy of the impurity  
                            else:
                                dat += [line]
                        
                dat += [':EORB   '+ str(IEorb)+'\n']
                dat += [':XEORB  '+ str(XEorb)+'\n']
                dat += [':ZEORB  '+ str(ZEorb)+'\n']
                fiorb=open('Eorb_imp.dat','w')
                fiorb.writelines(dat)  # Saves on the current directory the combination of all impurity problems
                fiorb.close()

                
                shutil.copy2('sig.inp','sig.inp.'+extn)
                sigmas.append('sig.inp.'+extn) # save the filename
                
            print('------- DMFT part of the step # ', itt, 'done ---------\n', file=fh_info)
            fh_info.flush()

            itt += 1   # END DMFT INNER LOOP
            
        n_imp *= 1./itt
        
        if int(len(sigmas)*p['saver'])>1 :
            sigmas.reverse()
            sigmas = sigmas[0:builtins.max(1,int(len(sigmas)*p['saver']))]
            print('Averaging over the following self-energies', sigmas, file=fh_info)
            cmd = 'saverage.py -o sig.inp ' + (' '.join(sigmas))
            print('#<saverage>: ', cmd, file=fh_info)
            subprocess.call(cmd,shell=True,stdout=fh_info,stderr=fh_info)

        (RelaxingStructure, mix_method) = fstc.AreWeRelaxingStructure2(w2k.case)
        if RelaxingStructure:
            fstc.TOTtoFOR(w2k.case)
            p['max_lda_iterations']=p['max_lda_iterations_optimize']
            print('Mixing method corresponds to structural optimization.', file=fh_info)
            print(' changing TOT to FOR in case.in2 files', file=fh_info)
            print(' increasing max_lda_iterations to ', p['max_lda_iterations'], file=fh_info)
        
        diff_nimp=1
        for ita in range(p['max_lda_iterations']):
            if ita%2 == 1 : p.refresh()

            recomputeEF = p['recomputeEF']

            Prepare_dmft2(dmfe, p, para, fh_info, recomputeEF, 'c')
            SSplit(dmfe.ROOT, inl.matsubara, p['iparams0'], p['ntail'], fh_info)
            dEF = dmft2(fday,w2k.case,fh_info,dmfe); fday.flush()
            

            if recomputeEF==0: # if we do not recomputeEF we need to make sure that density is correct, otherwise we have a doped compound!
                with open(w2k.case+'.outputdmf2') as f2:
                    lines = f2.readlines()
                drho=0
                for line in lines[::-1]:
                    if 'rho-rho_expected=' in line:
                        drho = float(line.split()[-1])
                        break
                print('rho-rho_expected=', drho, file=fh_info)
                if abs(drho)>0.1:
                    print('WARNING/ERROR: Since recomputeEF is set to 0, we expect EF in the gap.', file=fh_info)
                    print(' But density is wrong. Have rho-rho_expected=',drho, file=fh_info )
                    print(' You should adjust EF in EF.dat and see if EF falls into the gap', file=fh_info)
                    
            nd,Vdc = CreateEorb(w2k.case)

            nc = ConvertFromLatticeToImp(nd, imp2latt)

            lcore(fday,w2k.case,w2k.WIENROOT,fh_info);       fday.flush()
            scf  (w2k.case, {'cmplx': complex_suffix, 'so': '', 'm_extn': '', 'updn': ''})
            scf1 (fday,w2k.case,w2k.WIENROOT);
            mixer(fday,w2k.case,w2k.WIENROOT,fh_info);       fday.flush()
            scfm (fday,w2k.case,w2k.WIENROOT);               fday.flush()
            drho, dene = Diff(fday, w2k.case, dEF, diff_nimp, ddiff_nimp);fday.flush()
            
            # Writting to info.iterate file
            if os.path.isfile('EF.dat'): EF = float(open('EF.dat').read())
            (ETOT,SUM,XSUM,YSUM,ZSUM,EORB,XEORB,ZEORB) = FindEtot(w2k.case)
            
            fh_pinfo.write( '%3d %3s.%4s %12.6f %12.6f %15.6f %15.6f %15.6f ' % (istep0, icycle, ita, EF, Vdc, ETOT, ETOT-SUM+ZSUM-EORB+ZEORB, ETOT-SUM+YSUM-EORB+XEORB) )
            diff_nimp=0
            for iat in solver:   # Over all inequivalent impurity problems
                if os.path.exists('imp.'+str(iat)+'/Eimp.inp'):
                    fEimp = open('imp.'+str(iat)+'/Eimp.inp', 'r')
                    (Eimp,Olap,Edc) = [array([float(x) for x in line.split('#')[0].split()]) for line in fEimp.readlines()]  # throw out comments
                else: # If runGF=False, this is not available
                    Eimp=array([0,0])
                fh_pinfo.write( '%12.6f %12.6f %12.6f %12.6f ' % (nc[iat], n_imp[iat], Eimp[0], Eimp[-1]) )
                diff_nimp += n_imp[iat]-nc[iat]
            fh_pinfo.write('\n')
            fh_pinfo.flush()
            istep0 += 1
            print('------- LDA(charge) step', ita, icycle, 'done ---------', file=fh_info)
            
            istep1 = icycle+1
            istep2 = ita+1

            if fstc.AreWeRelaxingStructure(w2k.case):
                succi = False # Do not stop DFT iterations when relaxing structure
                if fstc.AreForcesConverged(w2k.case, how_many_times=3, min_Ene_diff=p['force_Ene_diff'], min_Rho_diff=p['force_Rho_diff'], info=fh_info):
                    fstc.StopStructureOptimization(w2k.case,info=fh_info)
            else:
                succi = (drho<p['cc'] and dene<p['ec'])
                
            if succi or ita==(p['max_lda_iterations']-1):
                istep1 += 1
                istep2 = 1
            
            print('\n:ITT%-2d   cycle %s \t%s %s/%s to go\n' % (istep2, istep1,time.asctime(),p['finish']-istep1,(p['finish']-istep1)%p['riter']), file=fday)
            
            lapw0(fday,w2k.case,w2k.WIENROOT,para, fh_info); fday.flush()
            lapw1(fday,w2k.case,w2k.WIENROOT,para,p['dftKS'],dmfe,complex_suffix,fh_info); fday.flush()
                
            fday.flush()
            
            if succi : break
            if ita>0 and ita%p['remove_broyd']==0 : 
                for f in glob.glob('*.broyd*'): 
                    os.remove(f)

        ddiff_nimp = abs(pdiff_nimp-diff_nimp)
        pdiff_nimp = diff_nimp
        
        drho, dene = Diff(fday, w2k.case, dEF, diff_nimp, ddiff_nimp);fday.flush()
        
        icycle += 1   # END CHARGE LOOP
        if RelaxingStructure: # When dmft/impurity loop restarts, we should relax again
            fstc.RelaxAgain(w2k.case, mix_method)
            shutil.copy2(w2k.case+'.struct',w2k.case+'.struct.'+str(icycle-1))
        
        if (p['max_lda_iterations']>=10): 
            toclean = glob.glob('*.broyd*')
            for f in toclean: os.remove(f)
        
        if not RelaxingStructure and ddiff_nimp<p['nc'] and drho<2*p['cc'] and dene<2*p['ec']:
            # impurity occupation is not changing much anymore, and charge and energy is converged, hence we can safely stop if structure is not relaxing.
            break
        
    toclean = glob.glob('dmft[1|2].error*')+glob.glob('*.broyd*')+glob.glob('dmft')+glob.glob('dmft2')+glob.glob('_processes*')+glob.glob('lapw1')+glob.glob('*.output1_*')+glob.glob('*.vector_*')+glob.glob('*.scf1_*')+glob.glob('lapw1_*.[def|error]')
    for f in toclean: os.remove(f)

    allFiles=glob.glob("*")
    for filename in allFiles:
        if os.stat(filename).st_size==0:
            os.remove(filename)
