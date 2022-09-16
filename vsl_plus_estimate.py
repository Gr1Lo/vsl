import warnings
warnings.filterwarnings("ignore")

#original work: https://www1.ncdc.noaa.gov/pub/data/paleo/softlib/vs-lite/

import numpy as np
m_ss = 1222
np.random.seed(m_ss)
from scipy.stats import zscore
import pandas as pd
import sys

def radians(deg):
    return deg * np.pi / 180

def estimate_vslite_params_v2_3(T,P,phi,RW,intwindow, nsamp, nbi, varargin=None):
    '''% Given calibration-interval temperature, precipitation, and ring-width data,
    % and site latitude, estimate_vslite_params_v2_3.m performes a Bayesian parameter
    % estimation of the growth response parameters T1, T2, M1 and M2. The
    % default prior distributions are based on a survey of current literature
    % pertaining to biological growth thresholds in trees; however uniform or
    % user-defined four-parameter-beta distributions may also optionally be
    % used as priors.  The scheme supports an assumption of either independent, 
    % Gaussian model errors, or AR(1) error structure, and in either case the
    % parameters of the error model may also be estimated.
    % 
    % For more details, see Tolwinski-Ward et al., 'Bayesian parameter estimation
    % and interpretation for an intermediate model of tree-ring width', Clim. Past, 
    % 9, 1-13, 3013, doi: 10.5194/cp-9-1-2013
    %
    % Basic Usage: [T1,T2,M1,M2] = estimate_vslite_params_v2_3(T,P,phi,RW)
    %
    % Basic Usage Inputs:
    % T: monthly calibration-interval temperature, dimension 12 x number of calibration years.
    % P: monthly calibration-interval precipitation, dimension 12 x number of cal. years.
    % phi: site latitude in degrees N.
    % RW: standardized annual calibration-interval ring-width index.
    %
    % Basic Usage Ouptuts:
    % T1, T2, M1, M2: point estimates given by the median of their respective
    %                 posterior distributions.
    %
    % Advanced Usage: [T1,T2,M1,M2,varargout] = vslite_bayes_param_cal(T,P,phi,RW,varargin)
    %
    % Advanced Optional Inputs:
    %     Must be specified as property/value pairs.  Valid property/value pairs are:
    %     'errormod'      Error model. Options are [0], [1], and [2] for white Gaussian
    %                     noise, AR(1) model, or AR(2) model.  Default is [0].
    %     'gparscalint'   Indices of years to use to estimate the growth response
    %                     parameters T1, T2, M1, M2. Default is all years.
    %     'eparscalint'   Indices of years to use to estimate the parameters of the
    %                     error model if these parameters are to be estimated.
    %                     Must be contiguous if using AR(1) error model. Default
    %                     is all years. (Note: may underestimate error if not disjoint
    %                     from interval used to fit growth response parameters
    %                     as specified in 'gparscalint'.)
    %     'errorpars'     Vector holding values of error model parameters is user
    %                     wishes to fix their values rather than estimate them.
    %                     For errormod == 0 (white noise model), values is a scalar
    %                     with fixed value for sigma2w; for errormod == 1 (AR(1)),
    %                     errorpars = [phi1 tau^2]; for errormod == 2 (AR(2)),
    %                     errorpars = [phi1 phi2 tau^2]. No default (since default is
    %                     to estimate parameters of a white noise error model).
    %     'pt_ests'       Choices are ['mle'] or ['med'] to return either the
    %                     posterior draw that maximizes the likelihood of the data
    %                     or the marginal posterior medians as the parameter point
    %                     estimates. Default is 'mle'.
    %     'hydroclim'     Is the hydroclimate input variable (2nd basic input) 
    %                     precipitation ['P'] or soil moisture ['M']? Default 
    %                     is ['P']; CPC Leaky Bucket model is then used to estimate 
    %                     M from input T and input P.
    %     'substep'       If hydroclim == 'P', then 'substep' is logical 0/1
    %                     depending on whether leaky bucket model without/with
    %                     substepping is preferred.  Default is [0].
    %     'intwindow'     VS-Lite integration window, specified as vector [I_0 I_f]
    %                     Default is [0 12].
    %     'nsamp'         200<=integer<=10,000 fixing number of MCMC iterations.
    %                     Default is [1000].
    %     'nbi'           Number of burn-in samples. Default is [200].
    %     'nchain'        Integer number of comp threads for the computation
    %                     of Rhat. Default is [3].
    %     'gparpriors'    Form of growth parameter priors.  Either can be
    %                     ['fourbet'] for a set of informative four-parameter 
    %                     beta-distributed priors, or can be ['uniform']. 
    %                     Parameterizations of these priors can be specified by
    %                     the user, or literature-based choices will be used as
    %                     defaults in either case (see following 5 property/value pairs
    %                     for more information). Defaults to ['fourbet'].
    %     'T1priorsupp'   2x1 vector with elements giving lower and upper bounds
    %                     for support of uniform T1 prior. If not included in input
    %                     argument list, default used is [0.0 8.5]
    %     'T2priorsupp'   " T2 prior. Default is [9.0 20.0]
    %     'M1priorsupp'   " M1 prior. Default is [0.01 0.03]
    %     'M2priorsupp'   " M2 prior. Default is [0.1 0.5]
    %     'fourbetparams' is a 4x4 matrix specifying parameters of the
    %                     four-parameter beta distributed priors.  First row
    %                     gives parameters of T1, second row gives T2, third
    %                     row gives parameters for M1, and 4th row params of
    %                     M2. Columns 1 and 2 give the two shape parameters,
    %                     while columns 3 and 4 give the lower and upper bounds
    %                     on the interval containing the transformed beta
    %                     distribution. If not included in input arguent list,
    %                     default parameter set based on current literature is
    %                               [9   5   0   9
    %                                3.5 3.5 10  24
    %                                1.5 2.8 0.0 0.1
    %                                1.5 2.5 0.1 0.5]
    %                     (See Tolwinski-Ward et al., doi: 10.5194/cp-9-1-2013,
    %                     for explanation of these choices.)
    %     'convthresh'    Scalar value greater than 0.  Threshold for MCMC
    %                     convergence; warning is displayed if abs(Rhat-1)>convthresh.
    %                     Default value is [0.1].
    %     'verbose'       Logical [0] or [1]; print progress to screen? Default 
    %                     is [1].
    %
    % Advanced Optional Ouptuts (must be specified in the following order):
    % T1dist, T2dist, M1dist, M2dist: Returns the entire numerical posterior distributions
    %                 of the growth response parameters if the user wants to check for
    %                 convergence, autocorrelation, multi-modality, etc., or to
    %                 use the full distributions of the parameters to quantify
    %                 uncertainty.
    % Rhats:          Returns the convergence statistics associated with T1, T2, M1, M2,
    %                 and sigma2rw if it was estimated.
    % convwarning:    Logical [0] or [1] depending on whether any Rhat values were
    %                 outside of the threshold distance from 1.
    % -- Next ordered outputs for white noise error model (errormod==0): -- %%%
    % sig2rw:         Point estimate of model error variance
    % sigma2rwdist:   Returns the entire numerical posterior distribution
    % Gdist:          Returns the numerical posterior distribution of monthly growth
    %                 responses to the input climate for the corresponding posterior
    %                 parameter sets; has dimension 12 x Nyrs x Nsamps.
    %                 
    % -- Next ordered outputs for AR(1) error model (errormod==1): -- %%%%%%%%%
    % phi1:           Point estimate of AR(1) coefficient
    % phi1dist:       Numerical posterior distribution of AR(1) coefficient 
    % tau2:           Point estimate of error model innovation variance
    % tau2dist:       Numerical distribution of error model innovation variance
    % Gdist:          Returns the numerical posterior distribution of monthly growth
    %                 responses to the input climate for the corresponding posterior
    %                 parameter sets; has dimension 12 x Nyrs x Nsamps.
    %
    % SETW 9/20/2011:  version 1.0. Estimates T1, T2, M1, M2 for fixed sigma2w, assuming
    %                  normally- and independent and identically distributed model residuals.
    % SETW 4/15/2012:  version 2.0. Added simultaneous estimation of sigma2w under assumption
    %                  of inverse-gamma prior, literature-based priors, Rhat convergence
    %                  metric, and output plots.
    % SETW 12/15/2012: version 2.1 for publication; added options for user-defined
    %                  prior distributions.
    % SETW 5/10/2013:  version 2.2: Revised data-level model and added options for white or
    %                  AR(1) error models following reviewer comments in
    %                  Climate of the Past Discussions; options for flexible
    %                  calibration-intervals for growth parameters and error
    %                  model parameters also added; MLE added as option for 
    %                  point estimates;
    %                  version 2.3: additional commenting added; option to
    %                  condition on user-supplied input soil moisture data included 
    %                  as opposed to necessarily estimating M from T & P via
    %                  Leaky Bucket.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''

    if varargin!=None: # read in advanced options if user-specified:
        #first fill values in with defaults:
        errormod = 0;
        '''gparscalint = 1:length(RW);
                    eparscalint = 1:length(RW);
                    pt_ests = 'mle';
                    hydroclim = 'P';
                    substep = 0;
                    intwindow = [0 12];
                    nsamp = 1000;
                    nbi = 200;
                    nchain = 3;
                    gparpriors = 'fourbet';
                    aT1 = 9; bT1 = (aT1+1)/2;
                    slpT1 = 9; intT1 = 0;
                    aT2 = 3.5; bT2 = aT2;
                    slpT2 = 14; intT2 = 10;
                    slpM1 = 0.1; intM1 = 0.0;
                    aM1 = 1.5; bM1 = (1-.035/slpM1)*aM1/(.035/slpM1); % .035 is target mean
                    slpM2 = 0.4; intM2 = 0.1;
                    aM2 = 1.5; bM2 = (1-(.25-intM2)/slpM2)*aM2/((.25-intM2)/slpM2); % .25 is target mean
                    convthresh = .1;
                    verbose = 1;
                    % then over-write defaults if user-specified:
                    Nvararg = length(varargin);
                    for i = 1:Nvararg/2
                        namein = varargin{2*(i-1)+1};
                        valin = varargin{2*i};
                        switch namein
                            case 'errormod'
                                errormod = valin;
                            case 'gparscalint'
                                gparscalint = valin;
                            case 'eparscalint'
                                eparscalint = valin;
                            case 'errorpars'
                                errorpars = valin;
                            case 'pt_ests'
                                pt_ests = valin;
                            case 'hydroclim'
                                hydroclim = valin;
                            case 'substep'
                                substep = valin;
                            case 'intwindow'
                                intwindow = valin;
                            case 'nsamp'
                                nsamp = valin;
                            case 'nbi'
                                nbi = valin;
                            case 'nchain'
                                nchain = valin;
                            case 'gparpriors'
                                gparpriors = valin;
                                if strcmp(gparpriors,'uniform')
                                    % read in default supports for uniform priors:
                                    aT1 = 0; bT1 = 9;
                                    aT2 = 10; bT2 = 24;
                                    aM1 = 0; bM1 = .1;
                                    aM2 = .1; bM2 = .5;
                                end
                            case 'T1priorsupp'
                                aT1 = valin(1); bT1 = valin(2);
                            case 'T2priorsupp'
                                aT2 = valin(1); bT2 = valin(2);
                            case 'M1priorsupp'
                                aM1 = valin(1); bM1 = valin(2);
                            case 'M2priorsupp'
                                aM2 = valin(1); bM2 = valin(2);
                            case 'fourbetparams'
                                aT1 = valin(1,1); bT1 = valin(1,2);
                                intT1 = valin(1,3); slpT1 = valin(1,4)-valin(1,3);
                                aT2 = valin(2,1); bT2 = valin(2,2);
                                intT2 = valin(2,3); slpT2 = valin(2,4)-valin(2,3);
                                aM1 = valin(3,1); bM1 = valin(3,2);
                                intM1 = valin(3,3); slpM1 = valin(3,4)-valin(3,3);
                                aM2 = valin(4,1); bM2 = valin(4,2);
                                intM2 = valin(4,3); slpM2 = valin(4,4)-valin(4,3);
                            case 'convthresh'
                                convthresh = valin;
                            case 'verbose'
                                verbose = valin;
                        end
                    end'''
    else : #otherwise, read in defaults:
        errormod = 0
        gparscalint = range(len(RW))
        eparscalint = range(len(RW))
        pt_ests = 'mle'
        hydroclim = 'P'
        substep = 0
        intwindow = [intwindow[0]-1, intwindow[1]-1]
        nsamp = nsamp #1000
        nbi = nbi #200
        nchain = 2 #3
        gparpriors = 'fourbet'
        aT1 = 9 
        bT1 = (aT1+1)/2
        slpT1 = 9
        intT1 = 0
        aT2 = 3.5
        bT2 = aT2
        slpT2 = 14
        intT2 = 10
        slpM1 = 0.1
        intM1 = 0.0
        aM1 = 1.5
        bM1 = (1-.035/slpM1)*aM1/(.035/slpM1) # .035 is target mean
        slpM2 = 0.4
        intM2 = 0.1
        aM2 = 1.5
        bM2 = (1-(.25-intM2)/slpM2)*aM2/((.25-intM2)/slpM2)# .25 is target mean
        convthresh = .1
        verbose = 1

    # Take zscore of RW data to fulfill assumptions of model error/noise structure
    RW = zscore(RW)
    Gterms_dist = np.empty((*T.shape, nsamp+nbi))

    # Compute soil moisture:
    Mmax =.76 # maximum soil moisture; v/v
    Mmin =.01 # minimum soil moisture; v/v
    muth = 5.8 # mu from thornthwaite's Ep scheme
    mth = 4.886 # m from thornthwaite's Ep scheme
    alpha = .093
    Minit = 200 # initial value for soil moisture; v/v
    dr = 1000 # root depth

    Nyrs = T.shape[1]

    # Read in or compute estimate of soil moisture M:
    if hydroclim=='P': #% if second input variable is precip,
        # then estimate soil moisture from T and P inputs via Leaky Bucket:
        if substep == 1:
            M = leakybucket_submonthly(1,Nyrs,phi,T,P,Mmax,Mmin,alpha,mth,muth,dr,Minit/dr)
        elif substep == 0:
            M = leakybucket_monthly(0,Nyrs,phi,T,P,Mmax,Mmin,alpha,mth,muth,dr,Minit/dr)

    elif hydroclim == 'M': # if user supplies soil moisture estimate from elsewhere,
        # read in the soil moisture from the second input variable:
        M = P

    # Compute monthly growth response to insolation, gE:
    gE = Compute_gE(phi)

    # Now do the MCMC sampling:
    for chain in range(nchain):
        #Storage space for realizations of parameters
        Tt = np.empty(nsamp+nbi)
        To = np.empty(nsamp+nbi)
        Mt = np.empty(nsamp+nbi)
        Mo = np.empty(nsamp+nbi)
        logLdata = np.empty(nsamp+nbi)
        
        if verbose: 
            print('Working on chain ' + str(chain) + ' out of ' + str(nchain) + '...')
        
        # Initialize the MCMC:
        gT = np.empty(T.shape)
        gM = np.empty(M.shape)
        sim = 0
        
        # Initialize growth response parameters:
        if gparpriors == 'uniform':
            # Initialize Tt and To with draws from priors:
            Tt[sim] = np.random.uniform(aT1,bT1)
            To[sim] = np.random.uniform(aT2,bT2)

            # Initialize Mt and Mo with draws from priors:
            Mt[sim] = np.random.uniform(aM1,bM1)
            Mo[sim] = np.random.uniform(aM2,bM2)

        elif gparpriors == 'fourbet':
            # Initialize Tt and To with draws from priors:
            Tt[sim] = slpT1*np.random.beta(aT1,bT1)+intT1
            To[sim] = slpT2*np.random.beta(aT2,bT2)+intT2

            # Initialize Mt and Mo with draws from priors:
            Mt[sim] = slpM1*np.random.beta(aM1,bM1)+intM1
            Mo[sim] = slpM2*np.random.beta(aM2,bM2)+intM2

        gT[T<Tt[sim]] = 0
        gT = np.where((T>Tt[sim]) & (T<To[sim]), (T-Tt[sim])/(To[sim]-Tt[sim]), gT)
        #gT[(T>Tt[sim]) and (T<To[sim])] = (T-Tt[sim])/(To[sim]-Tt[sim])
        
        gM[M<Mt[sim]] = 0
        gM[M>Mo[sim]] = 1
        #gM[(M>Mt[sim]) and (M<Mo[sim])] = (M[(M>Mt[sim]) and (M<Mo[sim])]-Mt[sim])/(Mo[sim]-Mt[sim])
        gM = np.where((M>Mt[sim]) & (M<Mo[sim]), (M-Mt[sim])/(Mo[sim]-Mt[sim]), gM)

        #??????
        Gterms = np.minimum(gM,gT) * np.repeat(gE,T.shape[1]).reshape(T.shape)
     
        sim = sim+1
        
        #================================================================================
        # The main sampling MCMC 
        # Create storage space to hold current values of the parameters to pass
        # to auxiliary sampling functions, and storage for MCMC realizations of these
        # parameters. Then initialize values of parameters.

        if errormod == 0:
            sigma2w = np.empty([nsamp+nbi,1])
            sigma2w[0] = np.random.uniform(0,1) # initialize estimate from the prior.
            errorpars = sigma2w[0] # errorpars holds current value of error model parameters.
        elif errormod == 1:
            tau2 = np.empty(nsamp+nbi,1)
            phi1 = np.empty(nsamp+nbi,1)

            # initialize estimates from the joint prior:
            phi1[0] = np.random.uniform(0,1)
            tau2[0] = np.random.uniform(0,1)
            while tau2[0] > 1-phi1[0]**2:
                phi1[0] = np.random.uniform(0,1)
                tau2[0] = np.random.uniform(0,1)
            
            # hold current values of error model parameters:
            errorpars[0] = phi1[0] 
            errorpars[1] = tau2[0]

        while (sim < nsamp+nbi): #+1
            #print('sim:'+ str(sim))

            if gparpriors == 'uniform':
                    Tt[sim] = TM_aux(Tt[sim-1],T,To[sim-1],gM,RW.T,errorpars,gE,Gterms,intwindow,gparscalint, 'U',aT1,bT1)
            if gparpriors == 'fourbet':
                    Tt[sim] = TM_aux(Tt[sim-1],T,To[sim-1],gM,RW.T,errorpars,gE,Gterms,intwindow,gparscalint,'L', aT1,bT1,slpT1,intT1)

            gT[T<Tt[sim]] = 0
            gT[T>To[sim-1]] = 1
            gT[(T>Tt[sim]) & (T<To[sim-1])] = (T[(T>Tt[sim]) & (T<To[sim-1])]-Tt[sim])/(To[sim-1]-Tt[sim])
            Gterms = np.minimum(gM,gT) * np.repeat(gE,T.shape[1]).reshape(T.shape)

            if gparpriors =='uniform':
                    To[sim] = TM_aux(To[sim-1],T,Tt[sim],gM,RW.T,errorpars,gE,Gterms,intwindow,gparscalint, 'U',aT2,bT2)
            if gparpriors =='fourbet':
                    To[sim] = TM_aux(To[sim-1],T,Tt[sim],gM,RW.T,errorpars,gE,Gterms,intwindow,gparscalint, 'L',aT2,bT2,slpT2,intT2)

            gT[T<Tt[sim]] = 0
            gT[T>To[sim]] = 1
            gT[(T>Tt[sim]) & (T<To[sim])] = (T[(T>Tt[sim]) & (T<To[sim])]-Tt[sim])/(To[sim]-Tt[sim])
            Gterms = np.minimum(gM,gT) * np.repeat(gE,T.shape[1]).reshape(T.shape)
            
            if gparpriors == 'uniform':
                    Mt[sim] = TM_aux(Mt[sim-1],M,Mo[sim-1],gT,RW.T,errorpars,gE,Gterms,intwindow,gparscalint, 'U',aM1,bM1)
            if gparpriors == 'fourbet':
                    Mt[sim] = TM_aux(Mt[sim-1],M,Mo[sim-1],gT,RW.T,errorpars,gE,Gterms,intwindow,gparscalint, 'L',aM1,bM1,slpM1,intM1);

            gM[M<Mt[sim]] = 0
            gM[M>Mo[sim-1]] = 1
            gM[(M>Mt[sim]) & (M<Mo[sim-1])] = (M[(M>Mt[sim]) & (M<Mo[sim-1])]-Mt[sim])/(Mo[sim-1]-Mt[sim])
            Gterms = np.minimum(gM,gT) * np.repeat(gE,T.shape[1]).reshape(T.shape)

            if gparpriors =='uniform':
                    Mo[sim] = TM_aux(Mo[sim-1],M,Mt[sim],gT,RW.T,errorpars,gE,Gterms,intwindow,gparscalint, 'U',aM2,bM2)
            if gparpriors == 'fourbet':
                    Mo[sim] = TM_aux(Mo[sim-1],M,Mt[sim],gT,RW.T,errorpars,gE,Gterms,intwindow,gparscalint, 'L',aM2,bM2,slpM2,intM2)

            gM[M<Mt[sim]] = 0
            gM[M>Mo[sim]] = 1
            gM[(M>Mt[sim]) & (M<Mo[sim])] = (M[(M>Mt[sim]) & (M<Mo[sim])]-Mt[sim])/(Mo[sim]-Mt[sim])
            Gterms = np.minimum(gM,gT) * np.repeat(gE,T.shape[1]).reshape(T.shape)

            Gterms_dist[:,:,sim] = Gterms 
            
            # Now draw from error model parameters:
            if errormod == 0:
                errorpars,logLdata[sim] = errormodel0_aux(errorpars,RW,Gterms,intwindow,eparscalint)
                sigma2w[sim] = errorpars

            elif errormod == 1:
                errorpars,logLdata[sim] = errormodel1_aux(errorpars,RW,Gterms,intwindow,eparscalint)
                phi1[sim] = errorpars[0]
                tau2[sim] = errorpars[1]

            sim = sim+1

        exec('global Ttchain{}; Ttchain{} = Tt'.format(str(chain),str(chain)))
        exec('global Tochain{}; Tochain{} = To'.format(str(chain),str(chain)))
        exec('global Mtchain{}; Mtchain{} = Mt'.format(str(chain),str(chain)))
        exec('global Mochain{}; Mochain{} = Mo'.format(str(chain),str(chain)))
        exec('global Gtermsdist{}; Gtermsdist{} = Gterms_dist'.format(str(chain),str(chain)))

        if errormod == 0:
            exec('global sig2rwchain{}; sig2rwchain{} = sigma2w'.format(str(chain),str(chain)))
        elif errormod == 1:
            exec('global phi1chain{}; phi1chain{} = phi1'.format(str(chain),str(chain)))
            exec('global tau2chain{}; tau2chain{} = tau2'.format(str(chain),str(chain)))

        exec('global logLchain{}; logLchain{} = logLdata'.format(str(chain),str(chain)))

        #print(logLdata)
    #================================================================================
    # POSTPROCESS SAMPLES:
    #================================================================================
    # assess convergence:
    Ttchains = []
    Tochains = []
    Mtchains = []
    Mochains = []
    Ttensemb = []
    Toensemb = []
    Mtensemb = []
    Moensemb = []
    Gterms_distensemb = []

    if errormod == 0:
        sig2rwchains = []
        sig2rwensemb = []
    elif errormod == 1:
        phi1chains = []
        phi1ensemb = []
        tau2chains = []
        tau2ensemb = []

    logLchains = []
    logLensemb = []

    for i in range(nchain):
        #???????????
        exec('Ttchains.append(Ttchain{})'.format(str(i)))
        exec('Tochains.append(Tochain{})'.format(str(i)))
        exec('Mtchains.append(Mtchain{})'.format(str(i)))
        exec('Mochains.append(Mochain{})'.format(str(i)))
        if errormod == 0:
                                        exec('sig2rwchains.append(sig2rwchain{}.T)'.format(str(i)))
        elif errormod == 1:
                                        exec('phi1chains.append(phi1chain{}.T)'.format(str(i)))
                                        exec('tau2chains.append(tau2chain{}.T)'.format(str(i)))

        exec('logLchains.append(logLchain{})'.format(str(i)))

        exec('Ttensemb.extend(Ttchain{}[nbi:])'.format(str(i)))
        exec('Toensemb.extend(Tochain{}[nbi:])'.format(str(i)))
        exec('Moensemb.extend(Mochain{}[nbi:])'.format(str(i)))
        exec('Mtensemb.extend(Mtchain{}[nbi:])'.format(str(i)))

        #????????????
        #exec('Gterms_distensemb = Gterms_distensemb.append(Gtermsdist{}[:,:,nbi+1:])'.format(str(chain)))

        if errormod == 0:
            exec('sig2rwensemb.extend(sig2rwchain{}[nbi:].T[0])'.format(str(i)))
        elif errormod == 1:
            exec('phi1ensemb.extend(phi1chain{}[nbi:].T[0])'.format(str(i)))
            exec('tau2ensemb.extend(tau2chain{}[nbi:].T[0])'.format(str(i)))

        exec('logLensemb.extend(logLchain{}[nbi:])'.format(str(i)))

    if (np.isnan(logLensemb).any()):
        print('=================================Chain contains Nan======================')
        return -9999999,-9999999,-9999999,-9999999,-9999999
    else:
      if pt_ests == 'med':
          T1 = np.median(Ttensemb)
          T2 = np.median(Toensemb)
          M1 = np.median(Mtensemb)
          M2 = np.median(Moensemb)
          if errormod == 0:
              sig2rw = np.median(sig2rwensemb)
          elif errormod == 1:
              phi1hat = np.median(phi1ensemb)
              tau2hat = np.median(tau2ensemb)

      elif pt_ests == 'mle':
          logLensemb = np.array(logLensemb)
          mle_ind = np.argwhere(logLensemb==np.max(logLensemb))[0][0]

          '''if len(mle_ind)>1:
                                      mle_ind = mle_ind[0]'''

          T1 = Ttensemb[mle_ind]
          T2 = Toensemb[mle_ind]
          M1 = Mtensemb[mle_ind]
          M2 = Moensemb[mle_ind]
          if errormod == 0:
              sig2rw = sig2rwensemb[mle_ind]
          elif errormod == 1:
              phi1hat = phi1ensemb[mle_ind]
              tau2hat = tau2ensemb[mle_ind]

      exec('global RhatT1; RhatT1 = gelmanrubin92(nsamp,nbi,Ttchains)')
      exec('global RhatT2; RhatT2 = gelmanrubin92(nsamp,nbi,Tochains)')
      exec('global RhatM1; RhatM1 = gelmanrubin92(nsamp,nbi,Mtchains)')
      exec('global RhatM2; RhatM2 = gelmanrubin92(nsamp,nbi,Mochains)')

      if errormod == 0:
          exec('global Rhatsig2rw;Rhatsig2rw = gelmanrubin92(nsamp,nbi,sig2rwchains)')
      elif errormod == 1:
          exec('global Rhatphi1;Rhatphi1 = gelmanrubin92(nsamp,nbi,phi1chains)')
          exec('global Rhattau2;Rhattau2 = gelmanrubin92(nsamp,nbi,tau2chains)')

      Rhats = [RhatT1, RhatT2, RhatM1, RhatM2]

      if verbose == 1:
          if errormod == 0:
              Rhats.append(Rhatsig2rw)
              print('    Rhat for T1, T2, M1, M2, sigma2rw:')
              print([RhatT1, RhatT2, RhatM1, RhatM2, Rhatsig2rw])
          elif errormod == 1:
              Rhats.append(Rhatphi1).append(Rhattau2)
              print('    Rhat for T1, T2, M1, M2, phi1, tau2:')
              print([RhatT1, RhatT2, RhatM1, RhatM2, Rhatphi1, Rhattau2])

      if np.any(np.abs(np.array(Rhats)-1)>convthresh):
          print('Gelman and Rubin metric suggests MCMC has not yet converged to within desired threshold;')
          print('Parameter estimation code should be re-run using a greater number of MCMC iterations.')
          print('(See ''nsamp'' advanced input option.)')
          convwarning = 1
      else:
          convwarning = 0

      return T1,T2,M1,M2,convwarning

#LEAKY BUCKET WITHOUT SUBSTEPPING 
def leakybucket_monthly(syear,eyear,phi,T,P,Mmax,Mmin,alph,m_th,mu_th,rootd,M0):
    '''
    % leackybucket_monthly.m - Simulate soil moisture with coarse monthly time step.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Usage: [M,potEv,ndl,cdays] = leakybucket_monthly(syear,eyear,phi,T,P,Mmax,Mmin,alph,m_th,mu_th,rootd,M0)
    %    outputs simulated soil moisture and potential evapotranspiration.
    %
    % Inputs:
    %   syear = start year of simulation.
    %   eyear = end year of simulation.
    %   phi = latitude of site (in degrees N)
    %   T = (12 x Nyrs) matrix of ordered mean monthly temperatures (in degEes C)
    %   P = (12 x Nyrs) matrix of ordered accumulated monthly precipitation (in mm)
    %   Mmax = scalar maximum soil moisture held by the soil (in v/v)
    %   Mmin = scalar minimum soil moisture (for error-catching) (in v/v)
    %   alph = scalar runoff parameter 1 (in inverse months)
    %   m_th = scalar runoff parameter 3 (unitless)
    %   mu_th = scalar runoff parameter 2 (unitless)
    %   rootd = scalar root/"bucket" depth (in mm)
    %   M0 = initial value for previous month's soil moisture at t = 1 (in v/v)
    %
    % Outputs:
    %   M = soil moisture computed via the CPC Leaky Bucket model (in v/v, 12 x Nyrs)
    %   potEv = potential evapotranspiration computed via Thornthwaite's 1947 scheme (in mm)
    %
    % SETW 2011
    '''

    #================================================================================
    iyear = list(range(syear, eyear))
    nyrs = len(iyear)
    #================================================================================

    # Storage for output variables (size [12 x Nyears]):
    M = np.empty((12,nyrs,))
    potEv = np.empty((12,nyrs,))

    # Compute normalized daylength (neglecting small difference in calculation for leap-years)
    latr = phi*np.pi/180  #change to radians
    ndays = [0,31,28,31,30,31,30,31,31,30,31,30,31]
    cdays = np.cumsum(ndays)
    sd = np.asmatrix(np.arcsin(np.sin(np.pi*23.5/180) * np.sin(np.pi * ((np.array(list(range(365))) + 1 - 80)/180)))).T #solar declination
    sd = np.asarray(sd).T
    y = -np.tan(np.ones(365) * latr) * np.tan(sd)
    y = np.where(y>=1,1,y)
    y = np.where(y<=-1,-1,y)

    hdl = np.arccos(y)
    dtsi = (hdl * np.sin(np.ones(365)*latr)*np.sin(sd))+(np.cos(np.ones(365) * latr) * np.cos(sd) * np.sin(hdl))
    ndl= dtsi[0]/max(dtsi[0]) # normalized day length

    # calculate mean monthly daylength (used for evapotranspiration in soil moisture calcs)
    jday = np.array(cdays[0:12]) + 0.5 * np.array(ndays[1:13]) #??????????

    m_star = 1-np.tan(radians(phi))*np.tan(radians(23.439*np.cos(jday*np.pi/182.625)))

    mmm = np.empty((12))
    for mo in range(12):
        if m_star[mo] < 0:
            mmm[mo] = 0
        elif m_star[mo] >0 and m_star[mo] < 2:
            mmm[mo] = m_star[mo]
        elif m_star[mo] > 2:
            mmm[mo] = 2

    nhrs = 24*np.degrees(np.arccos(1-mmm))/180; # the number of hours in the day in the middle of the month
    L = (np.array(ndays[1:13])/30)*(nhrs/12) # mean daylength in each month.

    # Pre-calculation of istar and I, using input T to compute the climatology:
    Tm=np.nanmean(T, axis=1)
    if len(Tm) !=12 :
        print('problem with creating T climatology for computing istar and I')

    elif len(Tm) ==12:
        istar = (Tm/5)**1.514 
        istar[Tm < 0] = 0;
        I=np.sum(istar)

    # precalculation of the exponent alpha in the Thornwaite (1948) equation:
    a = (6.75e-7)*(I**3) - (7.71e-5)*(I**2) + (1.79e-2)*I + .49;

    #================================================================================
    # -- year cycle -- 
    # syear = start (first) year of simulation
    # eyear = end (last) year of simulation
    # cyear = year the model is currently working on
    # iyear = index of simulation year

    for cyear in range(nyrs):     # begin cycling over years
        for t in range(12):  # begin cycling over months in a year
            # Compute potential evapotranspiration for current month after Thornthwaite
            if T[t,cyear] < 0:
                Ep = 0
            elif T[t,cyear]>=0 and T[t,cyear] < 26.5:
                Ep = 16*L[t]*(10*T[t,cyear]/I)**a
            elif T[t,cyear] >= 26.5:
                Ep = -415.85 + 32.25*T[t,cyear] - 0.43*(T[t,cyear]**2);

            potEv[t,cyear] = Ep

            # Now calculate soil moisture according to the CPC Leaky Bucket model (see J. Huang et al, 1996).
            if t > 0:
                # evapotranspiration:
                Etrans = Ep*M[t-1,cyear]*rootd/(Mmax*rootd)

                # groundwater loss via percolation:
                G = mu_th*alph/(1+mu_th)*M[t-1,cyear]*rootd

                # runoff; contributions from surface flow (1st term) and subsurface (2nd term)
                R = P[t,cyear]*(M[t-1,cyear]*rootd/[Mmax*rootd])**m_th + (alph/(1+mu_th))*M[t-1,cyear]*rootd
                dWdt = P[t,cyear] - Etrans - R - G;
                M[t,cyear] = M[t-1,cyear] + dWdt/rootd

            elif t == 0 and cyear > 0:
                # evapotranspiration:
                Etrans = Ep*M[11,cyear-1]*rootd/(Mmax*rootd)

                # groundwater loss via percolation:
                G = mu_th*alph/(1+mu_th)*M[11,cyear-1]*rootd

                # runoff; contributions from surface flow (1st term) and subsurface (2nd term)
                R = P[t,cyear]*(M[11,cyear-1]*rootd/(Mmax*rootd))**m_th +(alph/(1+mu_th))*M[11,cyear-1]*rootd
                dWdt = P[t,cyear] - Etrans - R - G
                M[t,cyear] = M[11,cyear-1] + dWdt/rootd

            elif t == 0 and cyear == 0:
                if M0 < 0:
                    M0 = .20

                # evapotranspiration (take initial soil moisture value to be 200 mm)
                Etrans = Ep*M0*rootd/(Mmax*rootd)

                # groundwater loss via percolation:
                G = mu_th*alph/(1+mu_th)*(M0*rootd)

                # runoff; contributions from surface flow (1st term) and subsurface (2nd term)
                R = P[t,cyear]*(M0*rootd/(Mmax*rootd))**m_th + (alph/(1+mu_th))*M0*rootd
                dWdt = P[t,cyear] - Etrans - R - G
                M[t,cyear] = M0 + dWdt/rootd

            # error-catching:
            if M[t,cyear] <= Mmin: 
                M[t,cyear] = Mmin

            if M[t,cyear] >= Mmax:
                M[t,cyear] = Mmax

            if np.isnan(M[t,cyear])==True:
                M[t,cyear] = Mmin

    return(M)

# SCALED DAYLENGTH
def Compute_gE(phi):
    '''% Just what it sounds like... computes just gE from latitude a la VS-Lite,
    % but without all the other stuff.
    %
    % Usage: gE = Compute_gE(phi)
    %
    % SETW 3/8/13'''

    gE = np.empty((12))

    # Compute normalized daylength (neglecting small difference in calculation for leap-years)
    latr = phi*np.pi/180  #change to radians
    ndays = [0,31,28,31,30,31,30,31,31,30,31,30,31]
    cdays = np.cumsum(ndays)
    sd = np.asmatrix(np.arcsin(np.sin(np.pi*23.5/180) * np.sin(np.pi * ((np.array(list(range(365))) + 1 - 80)/180)))).T #solar declination
    sd = np.asarray(sd).T
    y = -np.tan(np.ones(365) * latr) * np.tan(sd)
    y = np.where(y>=1,1,y)
    y = np.where(y<=-1,-1,y)

    hdl = np.arccos(y)
    dtsi = (hdl * np.sin(np.ones(365)*latr)*np.sin(sd))+(np.cos(np.ones(365) * latr) * np.cos(sd) * np.sin(hdl))
    ndl= dtsi[0]/max(dtsi[0]) # normalized day length

    # calculate mean monthly daylength (used for evapotranspiration in soil moisture calcs)
    jday = np.array(cdays[0:12]) + 0.5 * np.array(ndays[1:13]) #??????????
    m_star = 1-np.tan(radians(phi))*np.tan(radians(23.439*np.cos(jday*np.pi/182.625)))
    mmm = np.empty((12))
    for mo in range(12):
            if m_star[mo] < 0:
                mmm[mo] = 0
            elif m_star[mo] >0 and m_star[mo] < 2:
                mmm[mo] = m_star[mo]
            elif m_star[mo] > 2:
                mmm[mo] = 2

    #nhrs = 24*np.arccos(radians(1-mmm))/180 # the number of hours in the day in the middle of the month
    #L = (ndays[1:12]/30)*(nhrs/12) # mean daylength in each month.

    for t in range(12):
        gE[t] = np.mean(ndl[cdays[t]+1:cdays[t+1]])

    return gE


#================================================================================
# CONDITIONAL PARAMETER SAMPLING SUBROUTINES 
def TM_aux(mVar_curr,mVar,mVaro,gM,RW,errorpars,gE,Gterms,intwindow,cyrs,m_type,att,btt,slp=None,m_int=None):
    '''% gM/gT is the matrix of gM for all the years and all the months of the previous simulation.
    % T is the matrix of temperature for all years and all months of the previous simulation.
    % att is the lower bound on the support of the uniform prior distribution for Tt
    % btt is the upper bound on the support of the uniform prior distribution for Tt
    %
    % SETW 6/10/2010'''

    Ny = Gterms.shape[1]
    I_0 = intwindow[0] 
    I_f = intwindow[1]

    if m_type == 'U':
        mVar_prop = np.random.uniform(att,btt)
    else:
        mVar_prop = slp*np.random.beta(att,btt)+m_int

    gVarprop = np.ones([12,Ny])
    gVarprop[mVar<mVar_prop] = 0
    gVarprop[mVar>mVaro] = 1

    #gVarprop = np.where()
    gVarprop[(mVar<mVaro) & (mVar>mVar_prop)] = (mVar[(mVar<mVaro) & (mVar>mVar_prop)]-mVar_prop)/(mVaro-mVar_prop)

    #gprop0 = np.zeros((len(gE), len(gE)), float)
    #print(np.minimum(gM,gVarprop).shape)
    gprop = np.matmul(np.diag(gE), np.minimum(gM,gVarprop))
    gcurr = Gterms
        
    # account for variable integration window:
    if I_0<0:#if we include part of the previous year in each year's modeled growth:
            #??????????????
            startmo = 13+I_0 #13
            endmo = I_f

            prevseas = np.concatenate((np.mean(gprop[startmo:11,:],axis=1)[np.newaxis].T, gprop[startmo:11,:-1]), axis=1)
            gprop = gprop[:endmo,:]
            gprop = np.concatenate((prevseas, gprop))

            prevseas = np.concatenate((np.mean(gcurr[startmo:11,:],axis=1)[np.newaxis].T, gcurr[startmo:11,:-1]), axis=1)
            gcurr = gcurr[:endmo,:]
            gcurr = np.concatenate((prevseas, gcurr))


    else: # no inclusion of last year's growth conditions in estimates of this year's growth:
            startmo = I_0+1
            endmo = I_f
            gprop = gprop[startmo:endmo,:]
            gcurr = gcurr[startmo:endmo,:]

    if len(errorpars) == 1: #% White noise error model:
            sigma2rw = errorpars
            c0 = np.sqrt(1-sigma2rw)
            c1 = np.sum(gcurr[:,cyrs])-np.mean(np.sum(gcurr,axis=0))
            expcurr = np.sum((RW[cyrs].T - c0*c1/np.std(np.sum(gcurr,axis=0)))**2)

            p0 = np.sqrt(1-sigma2rw)
            p1 = np.sum(gprop[:,cyrs])-np.mean(np.sum(gprop,axis=0))
            expprop = np.sum((RW[cyrs].T - p0*p1/np.std(np.sum(gprop,axis=0)))**2)

            #?????????????????

            '''print(gprop)
            print(expprop)'''
            HR = np.exp(-.5*(expprop-expcurr)/sigma2rw)

    elif len(errorpars) == 2: # AR(1) error model:
            phi1 = errorpars[0]
            tau2 = errorpars[1]
            sigma2rw = tau2/(1-phi1**2)

            iSig = makeAR1covmat(phi1,tau2,len(cyrs))
            
            Wcurr = ((np.sum(gcurr[:,cyrs])-np.mean(np.sum(gcurr)))/np.std(np.sum(gcurr))).T
            Wprop = ((np.sum(gprop[:,cyrs])-np.mean(np.sum(gprop)))/np.std(np.sum(gprop))).T
            
            tprop = np.matmul(np.sqrt(1-sigma2rw),Wprop)
            logLprop = np.matmul(np.matmul(-.5*(RW[cyrs]-tprop).T,iSig),(RW[cyrs] - tprop))

            tcurr = np.matmul(np.sqrt(1-sigma2rw),Wcurr)
            logLcurr = np.matmul(np.matmul(-.5*(RW[cyrs]-tcurr).T,iSig),(RW[cyrs] - tcurr))
            HR = np.exp(logLprop-logLcurr)

    # accept or reject the proposal.

    if np.isnan(HR):
      return mVar_curr
    else:
      if np.random.binomial(1,min(HR,1))==1 : 
            return mVar_prop
      else:
            return mVar_curr



def errormodel0_aux(sigma2rwcurr,RW,Gterms,intwindow,cyrs):

    '''% RW = vector of observed annual ring widths
                % Gterms = vector of terms that sum together to give the simulated raw ring with index for all
                % months (rows) and all years (columns)
                % SETW 4/5/2013
                %'''

    # account for variable integration window:
    I_0 = intwindow[0] 
    I_f = intwindow[1]
    if I_0<0:# if we include part of the previous year in each year's modeled growth:
        startmo = 13+I_0
        endmo = I_f
        #prevseas = [np.mean(Gterms[startmo:11,:],axis=1),Gterms[startmo:11,:-1]]
        prevseas = np.concatenate((np.mean(Gterms[startmo:11,:],axis=1)[np.newaxis].T, Gterms[startmo:11,:-1]), axis=1)
        Gterms = Gterms[0:endmo,:]

        #?????????????
        #Gterms = [[prevseas], [Gterms]]
        gprop = np.concatenate((prevseas, Gterms))
    else: #% no inclusion of last year's growth conditions in estimates of this year's growth:
        startmo = I_0+1
        endmo = I_f
        Gterms = Gterms[startmo:endmo,:]

    # sample proposal from the prior:
    ################################################?????????????????????????/
    sigma2rwprop = (np.random.uniform(0,1))**2

    # accept or reject?
    Nyrs = len(cyrs)
    Gamma = np.squeeze(np.sum(Gterms,axis = 0))
    Gammabar = np.mean(Gamma)
    siggamma = np.std(Gamma)

    logprop = -.5*np.sum((RW[cyrs]-np.sqrt(1-sigma2rwprop)*(Gamma[cyrs]-Gammabar)/siggamma)**2)/sigma2rwprop
    logcurr = -.5*np.sum((RW[cyrs]-np.sqrt(1-sigma2rwcurr)*(Gamma[cyrs]-Gammabar)/siggamma)**2)/sigma2rwcurr
    HR = ((sigma2rwcurr[0]/sigma2rwprop)**(Nyrs/2))*np.exp(logprop-logcurr)

    if np.isnan(HR).any():
        HR = 1

    if np.random.binomial(1,min(HR,1))==1 : 
        sigma2rw = np.array([sigma2rwprop])
        logLdata = logprop-Nyrs/2*np.log(np.array([sigma2rwprop]))
    else:
        sigma2rw = sigma2rwcurr
        logLdata = logcurr-Nyrs/2*np.log(sigma2rwcurr)

    return sigma2rw,logLdata

def errormodel1_aux(currpars,RW,Gterms,intwindow,cyrs):
    '''% RW = vector of observed annual ring widths
    % Gterms = vector of terms that sum together to give the simulated raw ring with index for all
    % months (rows) and all years (columns)
    % SETW 4/5/2013'''

    # account for variable integration window
    I_0 = intwindow[0]
    I_f = intwindow[1]
    if I_0<0:# if we include part of the previous year in each year's modeled growth:
        startmo = 13+I_0
        endmo = I_f
        prevseas = [np.mean(Gterms[startmo:11,:],axis=1), Gterms[startmo:11,:-1]]
        Gterms = Gterms[:endmo,:]
        Gterms = [[prevseas], [Gterms]]
    else: #no inclusion of last year's growth conditions in estimates of this year's growth:
        startmo = I_0+1
        endmo = I_f
        Gterms = Gterms[startmo:endmo,:]

    # read current values of parameters:
    phi1curr = currpars[0]
    tau2curr = currpars[1]

    # if 0 % sample proposal from the prior:
    phi1prop = np.random.uniform(0,1)
    tau2prop = np.random.uniform(0,1)
    while tau2prop > 1-phi1prop**2:
        # satisfy conditions for stationarity, causality, and also
        # sigma2_w = tau2/(1-phi1^2) <= 1 since sigma2_w = 1/(1+SNR^2) in this model
        phi1prop = np.random.uniform(0,1)
        tau2prop = np.random.uniform(0,1)

    # accept or reject?
    Ny = len[cyrs]
    Gamma = np.squeeze(np.sum(Gterms))
    Gammabar = np.mean(Gamma)
    siggamma = np.std(Gamma)

    # VS-lite estimate of TRW at current parameter values:
    What = ((Gamma(cyrs)-Gammabar)/siggamma).T

    iSigprop,detSigprop = makeAR1covmat(phi1prop,tau2prop,Ny)
    iSigcurr,detSigcurr = makeAR1covmat(phi1curr,tau2curr,Ny)
    alphaprop = np.sqrt(1-tau2prop/(1-phi1prop**2))
    alphacurr = np.sqrt(1-tau2curr/(1-phi1curr**2))

    logLprop = -.5*np.matmul(np.matmul((RW[cyrs].T-alphaprop*What).T,iSigprop),(RW[cyrs].T-alphaprop*What))
    logLcurr = -.5*np.matmul(np.matmul((RW[cyrs].T-alphacurr*What).T,iSigcurr),(RW[cyrs].T-alphacurr*What))
    HR = np.sqrt(detSigcurr/detSigprop)*np.exp(logLprop-logLcurr)

    pars = np.ones(2)
    if np.random.binomial(1,min(HR,1))==1 : 
        pars[0] = phi1prop
        pars[1] = tau2prop
        logLdata = logLprop-np.log(detSigprop)
    else:
        pars[0] = phi1curr
        pars[1] = tau2curr
        logLdata = logLcurr-np.log(detSigcurr)

    return pars,logLdata



def makeAR1covmat(phi1,tau2,N):
    '''%%% [Sigma,invSigma,detSigma] = makeAR1covmat(phi1,tau2,N)
    % Make approximate joint covariance matrix, inverse covariace matrix, and covariance matrix
    % determinant for N sequential observations that follow the AR(1) model
    % X_t = phi1 X_t-1 + eps_t, where eps_t ~ N(0,tau^2)'''

    A = -phi1*np.identity(N)
    #???????????????????
    superdiag = np.ravel_multi_index([range(N-1)].T,[range(1,N)].T, dims=(N,N), order='F')
    A[superdiag] = 1

    invSigma = np.matmul((1/tau2)*(A.T), A)
    detSigma = (tau2/phi1**2)**N

    return invSigma,detSigma


def gelmanrubin92(Nsamp,Nbi,varargin):
    '''% Usage: Rhat = gelmanrubin92(Nsamp,Nbi,chain1,chain2,...,chainN)
    % Nsamp = number of iterations in each sample
    % Nbi = number to consider "burn-in".
    % chain1, ..., chainN must have dimension Nsamp x 1.
    % SETW 1/26/2011'''

    # Number of chains:
    m = len(varargin)

    if isinstance(varargin[0][0], float):
        for i in range(m):
            cv=0
            exec('global chain{};chain{}= varargin[i]'.format(str(i),str(i)))
    else:
        for i in range(m):
            cv=1
            exec('global chain{};chain{}= varargin[i][0]'.format(str(i),str(i)))
        

    # number of non-burn-in iterations:
    n = Nsamp-Nbi

    Xbar = np.empty(m)
    Xs2 = np.empty(m)
    allX = np.empty(n*m)
    for i in range(m):
        exec('global X;X = chain' + str(i))

        #within-chain means of X:
        Xbar[i] = np.nanmean(X[Nbi:len(X)])

        # within-chain variances of X:
        Xs2[i] = np.nanvar(X[Nbi:len(X)])
        allX[i*n:(i+1)*n] = X[Nbi:Nsamp]

    # mean across chains of mean X in each month:
    Xbarbar = np.mean(Xbar)

    BX = n*np.sum(((Xbar-Xbarbar)**2))/(m-1)

    WX = np.nanmean(Xs2)

    muhatX = np.nanmean(allX)

    sig2hatX = (n-1)*WX/n + BX/n; #% G&R92 eqn. 3

    VhatX = sig2hatX + BX/(m*n)
    ####??????
    varhatXs2 = np.var(Xs2,axis=0)

    covhatXs2Xbar2 = np.sum((Xs2-np.nanmean(Xs2))*(Xbar**2-np.nanmean(Xs2**2)))/m 
    covhatXs2Xbar = np.sum((Xs2-np.nanmean(Xs2))*(Xbar-np.nanmean(Xs2)))/m

    covhatXs2Xbar2 = covhatXs2Xbar2.T
    covhatXs2Xbar = covhatXs2Xbar.T

    varhatVhatX = (((n-1)/n)**2)*varhatXs2/m + (((m+1)/(m*n))**2)*2*BX**2/(m-1)+2*((m+1)*(n-1)/(m*n**2))*n*(covhatXs2Xbar2-2*Xbarbar*covhatXs2Xbar)/m

    dfX = 2*(VhatX**2)/varhatVhatX
    Rhat = (VhatX/WX)*(dfX/(dfX-2))

    return Rhat
    

#Simulate tree ring width index given monthly climate inputs
def VSLite_v2_5(syear,eyear,phi,T1,T2,M1,M2,T,P, intwindow, varargin=None):
    '''
                VSLite_v2_5.m - Simulate tree ring width index given monthly climate inputs.
                
                % Basic Usage:
                %    trw = VSLite_v2_5(syear,eyear,phi,T1,T2,M1,M2,T,P)
                %    gives just simulated tree ring as ouput.
                %
                %   [trw,gT,gM,gE,Gr,M] =
                %    VSLite_v2_5(syear,eyear,phi,T1,T2,M1,M2,T,P)) also includes
                %    growth response to temperature, growth response to soil moisture,
                %    scaled insolation index, overall growth function = gE*min(gT,gM),
                %    and soil moisture estimate in outputs.
                %
                % Basic Inputs:
                %   syear = start year of simulation.
                %   eyear = end year of simulation.
                %   phi = latitude of site (in degrees N)
                %   T1 = scalar temperature threshold below which temp. growth response is zero (in deg. C)
                %   T2 = scalar temperature threshold above which temp. growth response is one (in deg. C)
                %   M1 = scalar soil moisture threshold below which moist. growth response is zero (in v/v)
                %   M2 = scalar soil moisture threshold above which moist. growth response is one (in v/v)
                %     (Note that optimal growth response parameters T1, T2, M1, M2 may be estimated
                %      using code estimate_vslite_params_v2_5.m also freely available at
                %      the NOAA NCDC Paleoclimatology software library.)
                %   T = (12 x Nyrs) matrix of ordered mean monthly temperatures (in degEes C)
                %   P = (12 x Nyrs) matrix of ordered accumulated monthly precipitation (in mm)
                %
                % Advanced Inputs (must be specified as property/value pairs):
                %   'lbparams':  Parameters of the Leaky Bucket model of soil moisture.
                %                These may be specified in an 8 x 1 vector in the following
                %                order (otherwise the default values are read in):
                %                   Mmax: scalar maximum soil moisture content (in v/v),
                %                     default value is 0.76
                %                   Mmin: scalar minimum soil moisture (in v/v), default
                %                     value is 0.01
                %                   alph: scalar runoff parameter 1 (in inverse months),
                %                     default value is 0.093
                %                   m_th: scalar runoff parameter 3 (unitless), default
                %                     value is 4.886
                %                   mu_th: scalar runoff parameter 2 (unitless), default
                %                     value is 5.80
                %                   rootd: scalar root/"bucket" depth (in mm), default
                %                     value is 1000
                %                   M0: initial value for previous month's soil moisture at
                %                     t = 1 (in v/v), default value is 0.2
                %                   substep: logical 1 or 0; perform monthly substepping in
                %                     leaky bucket (1) or not (0)? Default value is 0.
                %   'intwindow': Integration window. Which months' growth responses should
                %                be intregrated to compute the annual ring-width index?
                %                Specified as a 2 x 1 vector of integer values. Both
                %                elements are given in integer number of months since January
                %                (July) 1st of the current year in the Northern (Southern)
                %                hemisphere, and specify the beginning and end of the integration
                %                window, respectively. Defaults is [1 ; 12] (eg. integrate
                %                response to climate over the corresponding calendar year,
                %                assuming location is in the northern hemisphere).
                %   'hydroclim': Value is a single character either taking value ['P'] or ['M'].
                %                If ['M'], then 9th input is interpreted as an estimate of
                %                soil moisture content (in v/v) rather than as precipitation.
                %                Model default is to read in precipitation and use the CPC's
                %                Leaky Bucket model of hydrology to estimate soil moisture,
                %                however if soil moisture observations or a more sophisticated
                %                estimate of moisture accounting for snow-related processes
                %                is available, then using these data directly are recommended
                %                (and will also speed up code).
                %
                % For more detailed documentation, see:
                % 1) Tolwinski-Ward et al., An efficient forward model of the climate
                % controls on interannual variation in tree-ring width, Climate Dynamics (2011)
                % DOI: 10.1007/s00382-010-0945-5
                %
                % 2) Tolwinski-Ward et al., Erratum to: An efficient forward model of the climate
                % controls on interannual variation in tree-ring width, Climate Dynamics (2011)
                % DOI: 10.1007/s00382-011-1062-9
                %
                % 3) Tolwinski-Ward et al., Bayesian parameter estimation and
                % interpretation for an intermediate model of tree-ring width, Clim. Past
                % (2013), DOI: 10.5194/cp-9-1-2013
                %
                % 4) Documentation available with the model at http://www.ncdc.noaa.gov/paleo/softlib/softlib.html
                %
                % Revision History
                % v0.1 - Original coding at monthly timestep from full daily timestep model (SETW, 4/09)
                % v1.0 - Changed soil moisture module to the CPC Leaky Bucket model (SETW, 5/09)
                % v1.1 - No upper parametric bounds for gT, gW as in full model; no density module (SETW, 9/09)
                % v1.2 - Added adjustable integration window parameters (SETW, 1/10)
                % v2.0 - Minor debugging for Octave compatibility, final version for publication (SETW, 10/10)
                % v2.1 - Error in evapotranspiration calculation corrected (SETW, 7/11)
                % v2.2 - Add switch to allow for monthly sub-stepping in soil moisture computation (SETW, N.Graham, K.Georgakaos, 9/11)
                % v2.3 - Add switch to allow moisture M to be given as input rather than estimated
                %        from T and P; add variable input options and improve
                %        commenting (SETW, 7/13)
                % v2.4 MNE debugged for using soil moisture inputs at l. 97-131
                % v2.5 Nick Graham (7/31/14) pointed out mistake in calculation of istar and
                % I at l. 350-352 and l. 484-486 in version 2.3, following Huang et al (1996), Equ. 3a:
                %
                %  i = (Tm/5) ** 1.514
                % 
                % - Here Tm is the climatological long-term monthly mean temperature
                % for month m - calculated over some suitable period. so there are
                % implicitly 12 values of i, and these are summed to give the
                % climatological value I for that site -
                %
                %  I = sum over 1-12 i(m)
                %
                % Note that this value could be calculated prior to beginning the
                % actual simulation.
                %
                % implemented by MNE (8/6/14).
                %
                % v2.5: also added Gr as an output, as varargout(5), and shuffling
                % varargout(4-8) to varargout(5-9). '''

    #================================================================================
    iyear = list(range(syear, eyear))
    nyrs = len(list(range(syear, eyear)))
    #================================================================================

    # Read in advanced inputs if user-specified; else read in parameter defaults:
    if varargin != None:
        # First fill parameter values in with defaults:
        # Parameters of the Leaky Bucket model:
        Mmax = 0.76
        Mmin = 0.01
        alph = 0.093
        m_th = 4.886
        mu_th = 5.80
        rootd = 1000
        M0 = 0.2
        substep = 0

        # Integration window parameters:
        I_0 = intwindow[0]-1
        I_f = intwindow[1]-1

        '''
        #Hydroclimate variable:
        hydroclim = 'P';
        for i = 1:nargin/2-1
            namein = varargin{i};
            switch namein
                case 'lbparams'
                    Mmax = varargin{i+1};
                    Mmin = varargin{i+2};
                    alph = varargin{i+3};
                    m_th = varargin{i+4};
                    mu_th = varargin{i+5};
                    rootd = varargin{i+6};
                    M0 = varargin{i+7};
                    substep = varargin{i+8};
                case 'intwindow'
                    I_0 = varargin{i+1};
                    I_f = varargin{i+2};
                case 'hydroclim'
                    hydroclim = varargin{i+1};
            end
        end'''
    else: #% otherwise, read in defaults:
        # Parameters of the Leaky Bucket model:
        Mmax = 0.76
        Mmin = 0.01
        alph = 0.093
        m_th = 4.886
        mu_th = 5.80
        rootd = 1000
        M0 = 0.2
        substep = 0

        # Integration window parameters:
        #I_0 = 1
        #I_f = 12

        I_0 = intwindow[0]-1
        I_f = intwindow[1]-1

        #Hydroclimate variable:
        hydroclim = 'P'

    #Pre-allocate storage for outputs:
    Gr = np.empty((12,nyrs,))
    gT = np.empty((12,nyrs,))
    gM = np.empty((12,nyrs,))
    M = np.empty((12,nyrs,))
    potEv = np.empty((12,nyrs,))

    #================================================================================
    # Load in or estimate soil moisture:
    if hydroclim == 'M':
        # Read in soil moisture:
        M = P
    else:
        #Compute soil moisture:
        '''if substep == 1;
                        M = leakybucket_submonthly(syear,eyear,phi,T,P,Mmax,Mmin,alph,m_th,mu_th,rootd,M0)'''
        if substep == 0:
            M = leakybucket_monthly(syear,eyear,phi,T,P,Mmax,Mmin,alph,m_th,mu_th,rootd,M0)
        elif substep !=1 and substep != 0:
            print('substep must either be set to 1 or 0.');
            return


    # Compute gE, the scaled monthly proxy for insolation:
    gE = Compute_gE(phi)

    #================================================================================
    # Now compute growth responses to climate, and simulate proxy:
    #================================================================================
    # syear = start (first) year of simulation
    # eyear = end (last) year of simulation
    # cyear = year the model is currently working on
    # iyear = index of simulation year

    # Compute monthly growth response to T & M, and overall growth response G:
    for cyear in range(nyrs): #begin cycling over years
        #================================================================================
        for t in range(12):  # begin cycling over months in a year
            # Calculate Growth Response functions gT(t) and gM(t)
            # First, temperature growth response:
            x = T[t,cyear]
            if (x < T1):
                gT[t,cyear] = 0
            elif (x >= T1) and (x <= T2):
                gT[t,cyear] = (x - T1)/(T2 - T1)
            elif (x >= T2):
                gT[t,cyear] = 1

            # Next, Soil moisture growth response:
            x = M[t,cyear]
            if (x < M1):
                gM[t,cyear] = 0
            elif (x >= M1) and (x <= M2):
                gM[t,cyear] = (x - M1)/(M2 - M1)
            elif (x >= M2):
                gM[t,cyear] = 1

        #================================================================================
        #????????????????
        # Compute overall growth rate:
        '''print(len(gE))
                                print(gT[:,cyear]) 
                                print(gM[:,cyear])
                                print(np.minimum(gT[:,cyear],gM[:,cyear]))'''
        Gr[:,cyear] = gE * np.minimum(gT[:,cyear],gM[:,cyear]);

    # Compute proxy quantity from growth responses 
    width = np.empty((nyrs))

    if phi>0: # if site is in the Northern Hemisphere:
        if I_0<0: # if we include part of the previous year in each year's modeled growth:
            startmo = 13 + I_0
            endmo = I_f

            # use average of growth data across modeled years to estimate first year's growth due
            # to previous year:

            '''width[0] = sum(Gr(1:endmo,1)) + sum(mean(Gr(startmo:12,:),2))
                            
                                    for cyear in range(nyrs):
                                        width(cyear) = sum(Gr(startmo:12,cyear-1)) + sum(Gr(1:endmo,cyear));'''
            
        else: # no inclusion of last year's growth conditions in estimates of this year's growth
            startmo = I_0; #+1???
            endmo = I_f;
            for cyear in range(nyrs):
                width[cyear] = np.sum(Gr[startmo:endmo,cyear])


    elif phi<0: #if site is in the Southern Hemisphere:
        # (Note: in the Southern Hemisphere, ring widths are dated to the year in which growth began!)
        startmo = 7+I_0 # (eg. I_0 = -4 in SH corresponds to starting integration in March of cyear)
        endmo = I_f-6 # (eg. I_f = 12 in SH corresponds to ending integraion in June of next year)
        for cyear in range(nyrs-1):
            width[cyear] = np.sum(Gr[startmo:12,cyear]) + np.sum(Gr[0:endmo,cyear+1])

        # use average of growth data across modeled years to estimate last year's growth due to the next year:
        width[nyrs-1] = np.sum(Gr[startmo:12,nyrs-1])+np.sum(np.mean(Gr[0:endmo,:])); #??????????

    trw = np.asmatrix(((width-np.mean(width))/np.std(width))).H #'; % proxy series is standardized width.
    trw = np.asarray(trw).T
    return(trw)

def estimate_and_compute_VSL(df, st_year, end_year,
                             tr_t_df_nn, tr_p_df_nn, t0_var_name, p0_var_name, tr_ret_lon, tr_ret_lat, u0, coor0,
                             t_df_nn, p_df_nn, t_var_name, p_var_name, ret_lon, ret_lat, u1, coor1,
                             min_lon = -145, min_lat = 14, max_lon = -52, max_lat = 71, nsamp=2000):
    lat_lon_list = []
    all_res = []
    cou = 0
    for fn in (df['file_name'].unique()):
      print(fn)
      df_t = df[df['file_name']==fn]
      df_t = df_t[df_t['age'].isin(tr_t_df_nn['year'].unique())]

      if len(df_t)>0:
        cou+=1
        if ((df_t['lat'].values[0] >= min_lat) & (df_t['lat'].values[0] <= max_lat) 
        & (df_t['lon'].values[0] >= min_lon) & (df_t['lon'].values[0] <= max_lon)):

          tr_lat_ind = (np.abs(df_t['lat'].values[0] - tr_ret_lat)).argmin()
          tr_lon_ind = (np.abs(df_t['lon'].values[0] - tr_ret_lon)).argmin()

          lat_ind = (np.abs(df_t['lat'].values[0] - ret_lat)).argmin()
          lon_ind = (np.abs(df_t['lon'].values[0] - ret_lon)).argmin()

          if 1:#[lat_ind, lon_ind] not in lat_lon_list:

              print('Расчет для точки ' +str(cou) + ' с индексом (CMIP_6) ' + str(lat_ind) + ' ' + str(lon_ind))

              tr_t_df_nn_t = tr_t_df_nn[(tr_t_df_nn[coor0[0]]==tr_ret_lon[tr_lon_ind]) 
              & (tr_t_df_nn[coor0[1]]==tr_ret_lat[tr_lat_ind]) 
              & (tr_t_df_nn['year'].isin(df_t['age'].unique()))]

              tr_p_df_nn_t = tr_p_df_nn[(tr_p_df_nn[coor0[0]]==tr_ret_lon[tr_lon_ind]) 
              & (tr_p_df_nn[coor0[1]]==tr_ret_lat[tr_lat_ind]) 
              & (tr_p_df_nn['year'].isin(df_t['age'].unique()))]

              t_for_pred = t_df_nn[(t_df_nn[coor1[0]]==ret_lon[lon_ind]) 
              & (t_df_nn[coor1[1]]==ret_lat[lat_ind])]
              p_for_pred = p_df_nn[(p_df_nn[coor1[0]]==ret_lon[lon_ind]) 
              & (p_df_nn[coor1[1]]==ret_lat[lat_ind])]

              t_df_nn_t = pd.pivot(tr_t_df_nn_t, index='year', columns="month", values=t0_var_name)
              p_df_nn_t = pd.pivot(tr_p_df_nn_t, index='year', columns="month", values=p0_var_name)
              t_for_pred = pd.pivot(t_for_pred, index='year', columns="month", values=t_var_name)
              p_for_pred = pd.pivot(p_for_pred, index='year', columns="month", values=p_var_name)

              if u0[0]=='degrees Celsius':
                T = t_df_nn_t.to_numpy().T
              else:
                T = t_df_nn_t.to_numpy().T-273.15

              if u0[1]=='mm/month':
                P = p_df_nn_t.to_numpy().T
              else:
                P = p_df_nn_t.to_numpy().T*86400*30.4167

              if u1[0]=='degrees Celsius':
                T_pred = t_for_pred.to_numpy().T
              else:
                T_pred = t_for_pred.to_numpy().T-273.15

              if u1[1]=='mm/month':
                P_pred = p_for_pred.to_numpy().T
              else:
                P_pred = p_for_pred.to_numpy().T*86400*30.4167

              np.set_printoptions(threshold=sys.maxsize)
              df_t0 = df_t[['age','trsgi']]
              df_t0.set_index('age', inplace=True)
              RW = df_t0.to_numpy().T[0]

              Tm=np.nanmean(T, axis=1)
              if len(Tm) ==12 :
                nsamp = nsamp#1000
                nbi = 200
                result = None
                
                while result is None:
                    try:
                        T0,T1,M0,M1, convwarning = estimate_vslite_params_v2_3(T,P,df_t['lat'].values[0],RW,[3,11],
                                                                              nsamp, nbi, varargin=None)
                        print(T0,T1,M0,M1, convwarning)
                        """
                        while convwarning == 1:
                              nsamp = nsamp * 20
                              #nbi = nbi * 2
                              print('nsamp = ' + str(nsamp))
                              T0,T1,M0,M1, convwarning = estimate_vslite_params_v2_3(T,P,df_t['lat'].values[0],RW,[3,11],
                                                                              nsamp, nbi, varargin=None)
                              print(T0,T1,M0,M1, convwarning)"""
                        result = 1
                    except:
                        pass

                if T0 != -9999999 and T_pred:
                  res = np.round(VSLite_v2_5(st_year,end_year,
                                            df_t['lat'].values[0],
                                            T0,T1,M0,M1,T_pred,P_pred,[3,11], varargin=None),3)
                
                  all_res.append(res)
                  lat_lon_list.append([lat_ind, lon_ind])
                  print(T0,T1,M0,M1)

    vsl_1000 = np.array(all_res).T#[:,0,:]
    return vsl_1000, lat_lon_list
