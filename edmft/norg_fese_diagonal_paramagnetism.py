#!/usr/bin/env python
# @Copyright 2007 Kristjan Haule
# 
import sys, os, glob, shutil
from numpy import *
from scipy import interpolate
from utils import DmftEnvironment
from cmpEimp3 import (
    DensityMatrix_nondiag,
    GiveImpFunctional_nondiag,
    SigmaInfinity_nondiag,
    fTrSigmaG_nondiag,
)
import subprocess
import dmft_DC
import builtins
import math
from numbers import Integral, Real as RealNumber

def green_solver_parameter(params):
    """Resolve g_mode/green_solver without changing the impurity parameters."""
    entries = []
    for key in ('g_mode', 'green_solver'):
        if key not in params:
            continue
        setting = params[key]
        if not isinstance(setting, (list, tuple)) or len(setting) != 2:
            raise ValueError("%s must be [-1, comment]" % key)
        value, comment = setting
        if isinstance(value, bool) or not isinstance(value, Integral) or value != -1:
            raise ValueError("%s supports only scalar Green Krylov mode -1" % key)
        entries.append((key, int(value), comment))
    if not entries:
        return None  # Keep the C++ default; do not force a mode here.
    return entries[0][1], entries[0][2]


def green_krylov_parameters(params):
    """Validate and resolve every supported GF option before any solver output.

    Zero g_mode_lanczos_step requests adaptive convergence. Positive values fix
    the basis budget; an explicit smaller safety cap must never truncate it.
    Return only requested options so C++ remains the owner of numerical defaults.
    """
    resolved = {}
    mode = green_solver_parameter(params)
    if mode is not None:
        resolved['green_solver'] = mode
    integer_bounds = {
        'g_mode_lanczos_step': (0, None),
        'green_krylov_max_basis': (0, None),
        'green_convergence_interval': (1, None),
        'green_convergence_stable_checks': (2, None),
        'green_check_level': (0, 3),
        'green_fallback': (0, 1),
        'lanczos_orth_method': (0, 1),
    }
    real_options = (
        'green_convergence_rtol', 'green_convergence_atol', 'green_check_tol',
        'green_deflation_tol', 'green_zero_tol',
    )
    for key in tuple(integer_bounds) + real_options:
        if key not in params:
            continue
        setting = params[key]
        if not isinstance(setting, (list, tuple)) or len(setting) != 2:
            raise ValueError("%s must be [value, comment]" % key)
        value, comment = setting
        if key in integer_bounds:
            lower, upper = integer_bounds[key]
            if (isinstance(value, bool) or not isinstance(value, Integral)
                    or value < lower or (upper is not None and value > upper)):
                raise ValueError("%s must be an integer in [%s, %s]" %
                                 (key, lower, 'infinity' if upper is None else upper))
            value = int(value)
        else:
            if (isinstance(value, bool) or not isinstance(value, RealNumber)
                    or not math.isfinite(value)):
                raise ValueError("%s must be a finite real number" % key)
            value = float(value)
            positive_fraction = key in (
                'green_convergence_rtol', 'green_check_tol', 'green_deflation_tol')
            if ((positive_fraction and not 0.0 < value < 1.0)
                    or (not positive_fraction and value < 0.0)):
                raise ValueError("Invalid tolerance %s" % key)
        resolved[key] = value, comment
    step = resolved.get('g_mode_lanczos_step', (0,))[0]
    cap = resolved.get('green_krylov_max_basis', (0,))[0]
    if step > 0 and cap > 0 and cap < step:
        raise ValueError("green_krylov_max_basis is smaller than fixed g_mode_lanczos_step")
    return resolved


def write_green_krylov_parameters(params, fp):
    """Write validated canonical GF options once, including convergence controls."""
    resolved = green_krylov_parameters(params)  # No partial write on invalid input.
    for key, (value, comment) in resolved.items():
        comment = ' '.join(str(comment).splitlines()).lstrip()
        if not comment.startswith('#'):
            comment = '# ' + comment
        print("%s  %s  %s" % (key, value, comment), file=fp)


def format_norg_param(v):
    """Recursively format a parameter for the bracket-based NORG input file."""
    if isinstance(v, ndarray):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        return '[' + ' '.join(format_norg_param(item) for item in v) + ']'
    return str(v)

def validate_diagonal_params(params):
    if 'magnetic' in params:
        setting = params['magnetic']
        if (not isinstance(setting, (list, tuple)) or len(setting) != 2
                or not isinstance(setting[0], Integral) or setting[0] != 0):
            raise ValueError("Only the legacy magnetic=[0, comment] is accepted.")
    if 'CoulombF' in params:
        setting = params['CoulombF']
        if (not isinstance(setting, (list, tuple)) or len(setting) != 2
                or setting[0] not in ("Ising", "'Ising'")):
            raise ValueError("This NORG solver supports only the Ising interaction.")
    if 'nondiag' in params or 'nondiag_blk' in params:
        raise ValueError("Remove nondiag: the diagonal orbital layout comes from Sigind.")
    if 'OffDiagonal' in params:
        raise ValueError("OffDiagonal is unavailable in the diagonal solver.")


def diagonal_layout(sigind):
    """Return active shell rows, their eDMFT component indices and local Sigind."""
    matrix = asarray(sigind)
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]
            or not issubdtype(matrix.dtype, integer)):
        raise ValueError("Sigind must be a square integer matrix.")
    labels = matrix.diagonal()
    if (matrix != diag(labels)).any() or (labels < 0).any():
        raise ValueError("NORG requires a diagonal Sigind with nonnegative entries.")
    rows = flatnonzero(labels)
    if len(rows) not in (2, 3, 5):
        raise ValueError("NORG requires 2, 3 or 5 active spatial orbitals.")
    # sjoin orders the independent positive eDMFT labels in ascending order.
    _, channels = unique(labels[rows], return_inverse=True)
    shifted = zeros_like(matrix)
    shifted[rows, rows] = channels + 1
    return rows, channels, shifted


def read_diagonal_file(filename, count):
    """Read a frequency followed by one real/imaginary pair per channel."""
    data = loadtxt(filename, ndmin=2)
    if data.shape[1] != 1 + 2 * count:
        raise ValueError("%s must have %d complex components" % (filename, count))
    return data[:, 0], data[:, 1::2] + 1j * data[:, 2::2]


def spin_diagonal(values):
    """Give the shared energy-functional routines both spins of every orbital."""
    values = repeat(asarray(values), 2, axis=-1)
    return values[..., :, None] * eye(values.shape[-1])


def read_occupations(filename, norbitals):
    occupancies = []
    total = None
    with open(filename) as stream:
        for line in stream:
            fields = line.split()
            if not fields or fields[0].startswith('#'):
                continue
            if fields[0] == 'sum':
                total = float(fields[1])
                break
            if fields[0] not in ('sup', 'sdn'):
                occupancies.append(float(fields[1]))
    if total is None or len(occupancies) != 2 * norbitals:
        raise ValueError("%s must contain paired spin occupancies and their total" % filename)
    return asarray(occupancies).reshape(norbitals, 2).sum(axis=1), total


def add_nf(self, params):
    _, sum_value = read_occupations(self.dir + 'nmat.txt', len(self.orbital_channels))
    # Step 2: Add nf to the beginning of Gf.out and Sig.out
    for filename in [self.dir + 'Gf.out', self.dir + 'Sig.out']:
        with open(filename, 'r') as file:
            data = file.read()
        with open(filename, 'w') as file:
            file.write("# nf=" + str(sum_value)  + '\n' + data)


Default_CoulombF = 'Ising'
mom=0       # We need those variables to be global in order to use python exect
TrSigmaG=0  # We need those variables to be global in order to use python exect
            
class IMP_NORG:
    """Wrapper for norg
    """
    def __init__(self, Znuc, dire, params, Sigind, CF):
        """Prepare the real-orbital, paramagnetic diagonal NORG interface."""
        validate_diagonal_params(params)
        green_krylov_parameters(params)
        self.active_rows, self.orbital_channels, self.Sigind_shifted = diagonal_layout(Sigind)
        self.nchannels = int(self.orbital_channels.max()) + 1
        # Environment
        self.env = DmftEnvironment()
        self.mpi_prefix = self.env.MPI
        
        # impurity directory
        self.dir = dire+'/'
        
        # create impurity directory if it does not yet exist
        #if len(glob.glob(dire))==0 : os.mkdir( dire )
        if not os.path.isdir(dire):
            os.mkdir( dire )

        self.Sigind = array(Sigind)
        self.CF = CF
            
        # log-file is opened
        self.fh_info = open(self.dir + 'info.out', 'w')

        # Note that run_dmft.py adds params['icase']<-indmfl.icase
        # and params['UlamJ'] (the latter only for exact DC)
        # to the params[iparams{xx}] from input file
        # 
        # CoulombF is needed later on, so we add one if not existant
        if 'CoulombF' not in params:
            params['CoulombF']=["'"+Default_CoulombF+"'", "# Full Coulomb repulsion"]
        # now all parameters in params are being remembered
        for p in params:
            if type(params[p][0]) is str:
                setattr(self, p, params[p][0])
            else:
                setattr(self, p, params[p][0])
            print('setting ctqmc.self.'+p+'=', params[p][0], file=self.fh_info)
        
        # Transformation between spheric harmonics and local coordinates needs to be
        # used in exact diagonalization of the atom.
        with open(self.dir + 'Trans.dat', 'w') as fh_T:
            print(len(Sigind), len(Sigind[0]), '#  size of Sigind and CF', file=fh_T)
            print('#---- Sigind follows', file=fh_T)
            max_sigfig = 1 + int(log(max(self.Sigind_shifted.flat))/log(10))
            fmt = '{:'+str(max_sigfig)+'d} '
            for row in self.Sigind_shifted:
                print((fmt*len(row)).format(*row), file=fh_T)
            print('#---- CF follows', file=fh_T)
            for row in CF:
                for elem in row:
                    print('{:12.8f} {:12.8f} '.format(elem.real,elem.imag), end=' ', file=fh_T)
                print(file=fh_T)
        
        # If old distribution of kinks exists, it is copied to imp directory
        status = glob.glob("status.*")
        if self.dir != './':
            print('Copying status files from to', self.dir, file=self.fh_info)
            for fil in status:
                shutil.copy2(fil, self.dir+'/'+fil)
        
        self.PARAMS = 'PARAMS.norg'
        
        self.sparams = {'exe': 'FeSe_diagonal_paramagnetism', 'Sig': 'Sig.out', 'Delta': 'Delta.inp',
                        'cix': 'actqmc.cix', 'Gf': 'Gf.out',
                        'nom': 100 if 'beta' not in params else int(3 * params['beta'][0])}

        # The driver supplies a spatial d shell. Standalone callers may confirm l explicitly.
        self.l = params.get('atom_l', params.get('l', [2]))[0]
        if self.l != 2 or len(Sigind) > 2 * self.l + 1:
            raise ValueError("NORG requires spatial d orbitals (l=2).")
        if 'UlamJ' in params and self.UlamJ[1] > 1e-5:
            self.J = self.UlamJ[3]

    def _exe(self, params, DCs, extn, UpdateAtom, gbroad=0.0, kbroad=0.0, maxAc=2000.):
        """Build diagonal solver inputs from the lattice components."""
        validate_diagonal_params(params)
        green_krylov_parameters(params)
        # Reading impurity levels
        (Eimp,Olap,Edc) = loadtxt(self.dir+'/Eimp.inp',ndmin=2)   # impurity levels should be produced in run_dmft.py at imp.x/Eimp.inp
        if len(Eimp) != self.nchannels:
            raise ValueError("Eimp.inp does not match the independent diagonal Sigind labels.")
        om, delta_channels = read_diagonal_file(self.dir + 'Delta.imp', self.nchannels)
        Delta = delta_channels[:, self.orbital_channels].T.copy()
        # subtracting Edc, because the ksum used (Hk+s_oo-Edc)
        #self.Eimps = Eimp-Edc                                     # impurity levels with double-counting included, as needed for ED & ctqmc
        self.Eimps = Eimp-Edc*Olap 

        if DCs=='fixn0':
            self.Eimps[:] = self.Eimps[self.orbital_channels].mean()
            print('Eimps(fixn0)=', self.Eimps, file=self.fh_info)
        
        ed = self.Eimps[self.orbital_channels]
        params['Ed'] = [ed.tolist(), "# impurity levels in physical orbital order"]
        params['Deg'] = [list(range(1, len(ed) + 1)), "# independent solver channels"]
        self.q_exe = self.env.ROOT + '/' + params.get('exe', [self.sparams['exe']])[0]
        
        # Compute nmesh (number of Matsubara frequencies in Delta.inp) for C++ solver
        T_mesh = 1./params['beta'][0]
        m_max_mesh = int(om[-1]/(pi*T_mesh)+1e-6)
        nmesh_val = len(arange(1, m_max_mesh+1, 2))
        params['nmesh'] = [nmesh_val, "# number of Matsubara frequencies in Delta.inp"]

        # Preparing params.dat file
        orbital_iteration_params = (
            'restrain_tight_divs',
            'norg_each_orbital_converg_error',
            'norg_each_orbital_converg_step_number',
        )
        with open(self.dir+self.PARAMS, 'w') as fp:
            write_green_krylov_parameters(params, fp)
            for p in ['Ed', 'Deg', 'J', 'CoulombF', 'beta', 'fit_range', 'fit_points', 'fit_nbaths', 'U', 'NOOC', 'restrain', 'restrain_difference', 'restrain_active_orbit_number', 'restrain_difference_G', 'restrain_active_orbit_number_G', 'control_divs_G', 'restrain_G_mode', 'restrain_autonooc', 'distribute', 'Minsulator', 'ful_pcl_sch', 'weight_nooc', 'weight_freze', 'nmesh', 'n_init', 'or_deg_adaptive', 'chi_A', 'chi_nomb', 'if_n_init', 'if_eigenstate_number', 'eigenstate_number', 'gs_lanczos_reorthogonalization', 'g_lanczos_reorthogonalization', 'testH0', 'testDelta', 'if_fix_N_subspace', 'fix_start_search_N_subspace', 'fix_N_subspace'] + list(orbital_iteration_params):
                if p in params:
                    if p == 'fix_N_subspace':
                        if len(params[p]) != 3:
                            raise ValueError(
                                "fix_N_subspace must be "
                                "[number_of_subspaces, subspaces, comment]"
                            )
                        number_of_subspaces, subspaces, comment = params[p]
                        if int(number_of_subspaces) != len(subspaces):
                            raise ValueError(
                                "fix_N_subspace declares %d subspaces but contains %d"
                                % (int(number_of_subspaces), len(subspaces))
                            )
                        value = [int(number_of_subspaces), subspaces]
                    else:
                        value, comment = params[p][0], params[p][1]
                    val_str = format_norg_param(value)
                    if p in orbital_iteration_params:
                        comment = str(comment).lstrip()
                        if not comment.startswith('#'):
                            comment = '# ' + comment
                    print("%s  %s " % (p, val_str), '\t ', comment, file=fp)
        shutil.copy2(self.dir+self.PARAMS, self.PARAMS)
        
        # All solver channels are diagonal and obey the causal sign condition.
        with open(self.dir+self.sparams['Delta'], 'w') as fp:
            Delta[abs(Delta) > maxAc*pi] = -1e-10j
            invalid = Delta.imag > 0
            Delta[invalid] = Delta[invalid].real - 1e-10j

            # creating big mesh
            T=1./params['beta'][0]
            m_max = int(om[-1]/(pi*T)+1e-6) # n of the last mastubara point
            lom = arange(1,m_max+1,2)*pi*T  # all Matsubata points included
            # interpolate on big mesh
            lDelta=[]
            
            print('shape(Delta)=', shape(Delta), file=self.fh_info)
            print('shape(om)=', shape(om), file=self.fh_info)
            print('shape(lom)=', shape(lom), file=self.fh_info)
            print('ss=', shape(Delta[0].real), file=self.fh_info)
            print('om[0]=', om[0], 'lom[0]=', lom[0], 'om[-1]=', om[-1], 'lom[-1]=', lom[-1], file=self.fh_info)
            
            for ib in range(len(Delta)):
                fDr = interpolate.UnivariateSpline(om, Delta[ib].real, s=0)
                fDi = interpolate.UnivariateSpline(om, Delta[ib].imag, s=0)
                lDelta.append( fDr(lom) + fDi(lom)*1j )
                
            for im,ome in enumerate(lom):
                print(ome, end=' ', file=fp)
                for b in range(len(lDelta)):
                    print("{:19.12f}  {:19.12f} ".format(lDelta[b][im].real, lDelta[b][im].imag), end=' ', file=fp)
                print(file=fp)
        
        shutil.copy2(self.dir+self.sparams['Delta'], self.dir+self.sparams['Delta']+'.'+extn)
        
        # saving three previous steps of status files, in case we need to restart from previous steps.
        if os.path.exists(self.dir+'status_2.tgz'):
            shutil.move(self.dir+'status_2.tgz', self.dir+'status_3.tgz')
        if os.path.exists(self.dir+'status_1.tgz'):
            shutil.move(self.dir+'status_1.tgz', self.dir+'status_2.tgz')
        if os.path.exists(self.dir+'status.000'):
            cmd = 'cd '+self.dir+'; tar czvf status_1.tgz status.*'
            ret = subprocess.call(cmd,shell=True,stderr=self.fh_info,stdout=self.fh_info)
        

        # Below we execute norg
        cmd = 'cd '+self.dir+'; '+self.mpi_prefix+' '+self.q_exe+' > nohup_imp.out 2>&1 '
        print(cmd, file=self.fh_info)
        print('Running ---- norg -----', file=self.fh_info)
        subprocess.check_call(cmd,shell=True,stdout=self.fh_info,stderr=self.fh_info)
        add_nf(self, params)

        
    def HighFrequency(self, params, DCs, nl_imp, extn, wbroad=0.0, kbroad=0.0):
        """ Reads impurity output and prepares for further execution
          Output:
               Sig[b][iom]  -- DMFT dynamic self-energy which vanishes at infinity
               sinf[b]      -- DMFT self-energy at infinity
               Edc[b]       -- double counting using DMFT occupancies
        """
        if kbroad>0.0 or wbroad>0.0:
            bexe = self.env.ROOT + '/broad'
            shutil.move(self.dir+self.sparams['Sig'], self.dir+self.sparams['Sig']+'b')
            broad_cmd = 'cd '+self.dir+'; '+bexe+' -w '+str(wbroad)+' -k '+str(kbroad)+' '+self.sparams['Sig']+'b  > '+self.sparams['Sig']
            print(broad_cmd, file=self.fh_info)
            subprocess.check_call(broad_cmd, shell=True, stdout=self.fh_info, stderr=self.fh_info)

        channels = self.orbital_channels
        om, physical_sigma = read_diagonal_file(self.dir + self.sparams['Sig'], len(channels))
        invalid = physical_sigma.imag > 0
        physical_sigma[invalid] = physical_sigma[invalid].real
        # The lattice consumes one self-energy per label; retain physical columns for energy.
        Sigo = array([physical_sigma[:, channels == label].mean(axis=1)
                      for label in range(self.nchannels)])
        sinf = Sigo[:, -1].real.copy()

        global mom, nf, ntot, TrSigmaG
        orbital_occupations, nf = read_occupations(self.dir + 'nmat.txt', len(channels))
        # ComputeExactDC divides each channel occupation by its orbital multiplicity.
        mom = bincount(channels, weights=orbital_occupations, minlength=self.nchannels)
        ntot = nf
        print('From impurity Sigma: nf=', nf, 'sinf=', sinf, file=self.fh_info)

        # Reading old Edc and impurity levels
        (Eimp_old,Olap_old,Edc_old) = loadtxt(self.dir+'/Eimp.inp',ndmin=2)
        # Keep the DC that was used to generate the current NORG G and Sigma.
        # The newly computed Edc below belongs to the next DMFT iteration.
        Edc_solver = copy(Edc_old)
        
        print('len(Edc_old)=', len(Edc_old), 'Eimp_old=', Eimp_old, file=self.fh_info)

        #if 'CoulombF' in params:
        #    CoulombF = params['CoulombF'][0].strip("'").strip('"')
        #else:
        #    CoulombF = Default_CoulombF
        
        # Here we compute self-energy(infinity) and 
        # double-counting using impurity occupancy
        print('DCs=', DCs, file=self.fh_info)
        Ud = params['U'][0]
        if type(self.J) is list:
            Jd = sum(self.J)/len(self.J)
        else:
            Jd = self.J
        # If we want to have U different than the U used in double counting
        # (useful for cluster-DMFT when U can not be simply added in ctqmc)
        if 'Ud' in params: Ud = params['Ud'][0]
        if 'Jd' in params: Jd  = params['Jd'][0]
        
        DC_ones = ones_like(Edc_old)

        if DCs[:5]=='exact':
            # We allow DCs = ['exact', 'exact_l', 'exactd', 'exactdl']
            # exact and exactd use impurity occupation. exact uses yukawa&dielectric, while exactd uses dielectric screening only
            # exact_l and exactdl use lattice occupations. exact_l uses yukawa&dielectric, while exactdl uses dielectric screening only
            lmbda = self.UlamJ[1]
            epsilon = self.UlamJ[2]
            nf_ = mom
            print('nl_imp=', nl_imp, file=self.fh_info)
            print('Here we start n_imp=', nf_, 'isscalar(nlattice)=', isscalar(nl_imp), file=self.fh_info)
            if len(DCs)>6 and DCs[6]=='l' and not isscalar(nl_imp): 
                nf_ = nl_imp
            print('Here we have nl_imp=', nf_, file=self.fh_info)
            print('lmbda=', lmbda, 'epsilon=', epsilon, file=self.fh_info)
            (Edc0,Phidc0)=dmft_DC.ComputeExactDC(nf_, lmbda, epsilon, self.icase, self.fh_info, projector='projectorw.dat',trans=self.dir+'Trans.dat')
            print('The Exact Double-counting is: Vdc=', Edc0, 'PhiDC=', Phidc0, file=self.fh_info)

        elif DCs=='FLL' or DCs=='default' or DCs=='FLL_l':
            nf_ = nf
            if DCs=='FLL_l' and nf_: 
                nf_ = sum(nl_imp)
            # This is the fully localized limit double-counting by Anisimov
            Edc0 = DC_ones*( Ud*(nf_-0.5) - Jd*(nf_/2.-0.5) )
            Phidc0 = Ud*0.5*nf_*(nf_-1.) - Jd*0.25*nf_*(nf_-2.)
        elif DCs=='AMF':
            nf_ = nf
            if DCs=='AMF_l' and nf_: 
                nf_ = sum(nl_imp)
            Ueff = Ud*(1-1./(2.*(2.*self.l+1.))) - Jd*self.l/(2*self.l+1.)
            Edc0 = DC_ones*Ueff*nf_
            Phidc0 = Ueff*nf_**2/2.
        elif DCs=='fixn' or DCs=='nominal':         # If the scheme if fixn, we keep double-counting potential fixed
            nf0 = params['nf0'][0]
            Vdc = Ud*(nf0-0.5) - Jd*(nf0/2.-0.5)
            Edc0 = DC_ones * Vdc
            Phidc0 = ntot*Vdc
        else:
            print('ERROR: Unkown double-counting', DCs)
            sys.exit(1)
            
        if 'dDC' in params:
            Edc0 += array(params['dDC'][0])
            density_for_shift = nf_ if DCs.startswith('exact') else mom
            Phidc0 += dot(asarray(params['dDC'][0]), density_for_shift)
            print('The Double-counting corrected to: Vdc=', Edc0, 'PhiDC=', Phidc0, file=self.fh_info)

        Edc = copy(Edc_old) # we want to keep Edc<-1000 for orbitals which are projected out!
        for i in range(len(Edc)):
            if (Edc_old[i]>-1000.): # this is true for all orbitals which are kept!
                Edc[i] = Edc0[i]
            else:                   # these orbitals are projected out -> we want to keep sinf=Edc for orbitals projected out!
                sinf[i] = Edc[i]
                

        # Copying data at each iteration to follow the evolution
        shutil.copy2(self.dir+self.sparams['Sig'], self.dir+self.sparams['Sig']+'.'+extn)
        shutil.copy2(self.dir+self.sparams['Gf'], self.dir+self.sparams['Gf']+'.'+extn)
        shutil.copy2(self.dir+'PARAMS.norg', self.dir+'PARAMS.norg'+'.'+extn)
        shutil.copy2(self.dir+'nohup_imp.out', self.dir+'norg.log'+'.'+extn)
        if os.path.exists(self.dir+'hb_fit.txt'): shutil.copy2(self.dir+'hb_fit.txt', self.dir+'hb_fit'+'.'+extn)
        if os.path.exists(self.dir+'ose_hop'): shutil.copy2(self.dir+'ose_hop', self.dir+'ose_hop'+'.'+extn)
        if os.path.exists(self.dir+'nmat.txt'): shutil.copy2(self.dir+'nmat.txt', self.dir+'nmat'+'.'+extn)
        if os.path.exists(self.dir+'h0.txt'): shutil.copy2(self.dir+'h0.txt', self.dir+'h0'+'.'+extn)
        
        # The shared functional takes full spin matrices, in physical orbital order.
        omega_g, green = read_diagonal_file(self.dir + self.sparams['Gf'], len(channels))
        omega_delta, delta = read_diagonal_file(self.dir + self.sparams['Delta'], len(channels))
        if len(omega_g) != len(om) or not allclose(omega_g, om, rtol=1e-9, atol=1e-12):
            raise ValueError('Gf.out and Sig.out must use the same frequency mesh')
        G_matrices = spin_diagonal(green)
        Delta_matrices = spin_diagonal(delta)
        Sigma_matrices = spin_diagonal(physical_sigma)
        local_energy_matrix = spin_diagonal((Eimp_old - Edc_solver * Olap_old)[channels])
        Vdc_matrix = spin_diagonal(Edc[channels])

        density_matrix = DensityMatrix_nondiag(
            omega_g, G_matrices
        )
        sigma_infinity_matrix = SigmaInfinity_nondiag(
            Sigma_matrices, om
        )
        TrSigmaG_full = fTrSigmaG_nondiag(
            omega_g,
            G_matrices,
            Sigma_matrices,
            density_matrix=density_matrix,
            sigma_infinity=sigma_infinity_matrix,
            fout=self.fh_info,
        )
        # Historical Eorb.dat uses TrSigmaG for the interaction energy,
        # i.e. one half of the full Matsubara trace.
        TrSigmaG = 0.5 * TrSigmaG_full

        (
            Phi_DMFT,
            lnGimp,
            Fimp,
            Vdc_nd,
            zeorb,
        ) = GiveImpFunctional_nondiag(
            omega_g,
            G_matrices,
            Sigma_matrices,
            Delta_matrices,
            local_energy_matrix,
            Vdc_matrix,
            mu=0.0,
            omega_delta=omega_delta,
            nf=ntot,
            sigma_infinity=sigma_infinity_matrix,
            fout=self.fh_info,
        )

        print(
            'nimp, TrSigmaG/2=',
            ntot,
            TrSigmaG,
            'Vdc*nd=',
            Vdc_nd,
            file=self.fh_info,
        )
        self.fh_info.flush()

        Ry2eV = 13.60569193
        Epotential = TrSigmaG
        Phidc_value = float(asarray(Phidc0).sum())

        with open(self.dir+'/Eorb.dat', 'w') as fE:
            print('#  Phidc=', Phidc_value, file=fE)
            print('#  Tr(Sigma*G)/2=', Epotential, file=fE)
            print(
                '#  Tr(Sigma*G)/2-Phidc=',
                Epotential-Phidc_value,
                file=fE,
            )
            print('#  Phi_DMFT=', Phi_DMFT, file=fE)
            print(
                '#  Phi_DMFT-Phi_DC=',
                Phi_DMFT-Phidc_value,
                file=fE,
            )
            print('#  Tr(log(-Gimp))=', lnGimp, file=fE)
            print('#  Fimpurity(T=0)=', Fimp, file=fE)
            print(
                '#  E_pot+E_kin-Tr(w*dD/dw)=',
                zeorb,
                file=fE,
            )
            print(
                ':IEORB ',
                (Epotential-Phidc_value)/Ry2eV,
                file=fE,
            )
            print(
                ':XEORB ',
                (Phi_DMFT-Phidc_value)/Ry2eV,
                file=fE,
            )
            print(
                ':ZEORB ',
                (zeorb-Phidc_value)/Ry2eV,
                file=fE,
            )

        with open(self.dir+'/sig.out', 'w') as fE:
            print('# s_oo=', sinf.tolist(), file=fE)
            print('# Edc=', Edc.tolist(), file=fE)
            for iom in range(len(om)):
                print(("%20.15f " % om[iom]), end=' ', file=fE)
                for b in range(len(Sigo)):
                    print(("%20.15f %20.15f  " % (Sigo[b,iom].real-Edc[b], Sigo[b,iom].imag)), end=' ', file=fE)
                print(file=fE)
        
        return ntot

def ferm(x):
    """Fermi function for T=1"""
    if (x>100): return 0
    if (x<-100): return 1
    return 1/(exp(x)+1)


if __name__ == '__main__':
    from readTrans import ReadTrans
    iparams0={"exe"     : ["FeSe_diagonal_paramagnetism", "# Name of the executable"],
          "U"           : [5.0             , "# Coulomb repulsion (F0)"],
          "Ud"          : [5.0             , "# Coulomb repulsion (F0)"],
          "J"           : [0.8             , "# Coulomb repulsion (F0)"],
          "nf0"         : [1               , "# Double counting parameter"],
          "beta"        : [35.0            , "# Inverse temperature"],
          "Nmax"        : [2000            , "# Maximum perturbation order allowed"],
          "M"           : [1e8             , "# Total number of Monte Carlo steps"],
          "Ncout"       : [1000000         , "# How often to print out info"],
          "Naver"       : [1000000000      , "# How often to print out debug info"],
          "nom"         : [50              , "# Number of Matsubara frequency points sampled"],
          "aom"         : [4               , "# Number of frequency points used to determin the value of sigma at nom"],
          "sderiv"      : [0.02            , "# Maximum derivative mismatch accepted for tail concatenation"],
          "Ntau"        : [1000            , "# Number of imaginary time points (only for debugging)"],
          "SampleGtau"  : [1000            , "# How often to update G(tau)"],
          "GlobalFlip"  : [500000          , "# How often to try a global flip"],
          "tsample"     : [10              , "# How often to record measurements"],
          "warmup"      : [100000          , "# Warmup number of QMC steps"],
          "CleanUpdate" : [100000          , "# How often to make clean update"],
          "minM"        : [1e-10           , "# The smallest allowed value for the atomic trace"],
          "minD"        : [1e-10           , "# The smallest allowed value for the determinant"],
          "PChangeOrder": [0.9             , "# Ratio between trial steps: add-remove-a-kink / move-a-kink"],
          "CoulombF"    : ["'Ising'"        , "# Georges = rough run"],
          "atom_OCA_G"  : [False           , "# Don't compute OCA diagrams for speed"],
          "UlamJ"       : [[5.0, 1.360508, 1.4459, [0.7506725, 0.8493274]], "#"],
          "icase"       : [0               , "# added by run_dmft.py"],
    }

    
    dire = 'imp.0'
    (Sigind, CF) = ReadTrans(dire+'/Trans.dat', sys.stdout)

    imp = IMP_NORG(23, dire, iparams0, Sigind, CF)
    imp._exe(iparams0, 'exact', '_bris', 1)
    
    ntot=imp.HighFrequency(iparams0, 'exact', 0.0, '_bris', 0.05, 0.1)
    print('ntot=', ntot)


