#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, unicode_literals, print_function
import pymultinest
import os
try: os.mkdir('chains')
except OSError: pass

import numpy as np
import matplotlib.pyplot as pl
import sys
sys.path.append('/home/diwen2/scratch/castor')
import visilens as vl
import corner
import scipy.sparse
from astropy.cosmology import Planck15

plotfbase = 'CLASS_B1422+231_modelcont'

datasets = []
datasets.append(vl.read_visdata('../B1422_X534_selfcal_timebin6.bin'))
datasets.append(vl.read_visdata('../B1422_X30db_selfcal_timebin6.bin'))
data = vl.concatvis(datasets)
print(data.real.shape)

xmax = 12.5
highresbox = [-2.5, +1.5, -2, +2] # xmin, xmax, ymin, ymax
fieldres, emitres = 0.05, 0.025

scaleamp =   False # do not allow flux re-scaling
shiftphase = False # do not allow astrometric shift

modelcal = True

# Making these lists just makes later stuff easier since we now know the dtype
#lens = list(np.array([lens]).flatten())
#source = list(np.array([source]).flatten()) # Ensure source(s) are a list
data = list(np.array([data]).flatten())     # Same for dataset(s)
scaleamp = list(np.array([scaleamp]).flatten())
shiftphase = list(np.array([shiftphase]).flatten())
modelcal = list(np.array([modelcal]).flatten())
if len(scaleamp)==1 and len(scaleamp)<len(data): scaleamp *= len(data)
if len(shiftphase)==1 and len(shiftphase)<len(data): shiftphase *= len(data)
if len(modelcal)==1 and len(modelcal)<len(data): modelcal *= len(data)
sourcedatamap = [None]*len(data)

# Get any model-cal parameters set up. The process involves some expensive
# matrix inversions, but these only need to be done once, so we'll do them
# now and pass the results as arguments to the likelihood function. See docs
# in calc_likelihood.model_cal for more info.
for i,dset in enumerate(data):
    if modelcal[i]:
        uniqant = np.unique(np.asarray([dset.ant1,dset.ant2]).flatten())
        dPhi_dphi = np.zeros((uniqant.size-1,dset.u.size))
        for j in range(1,uniqant.size):
            dPhi_dphi[j-1,:]=(dset.ant1==uniqant[j])-1*(dset.ant2==uniqant[j])
        C = scipy.sparse.diags((dset.sigma/dset.amp)**-2.,0)
        F = np.dot(dPhi_dphi,C*dPhi_dphi.T)
        Finv = np.linalg.inv(F)
        FdPC = np.dot(-Finv,dPhi_dphi*C)
        modelcal[i] = [dPhi_dphi,FdPC]

# Create our lensing grid coordinates now, since those shouldn't be
# recalculated with every call to the likelihood function
xmap,ymap,xemit,yemit,indices = vl.GenerateLensingGrid(data,xmax,highresbox,fieldres,emitres)

# Calculate the uv coordinates we'll interpolate onto; only need to calculate
# this once, so do it here.
arcsec2rad = np.pi/180/3600
kmax = 0.5/((xmap[0,1]-xmap[0,0])*arcsec2rad)
ug = np.linspace(-kmax,kmax,xmap.shape[0])

# Calculate some distances; we only need to calculate these once.
# This assumes multiple sources are all at same z; should be this
# way anyway or else we'd have to deal with multiple lensing planes
cosmo = Planck15
lens_z = 0.34
source_z = 3.62
Dd = cosmo.angular_diameter_distance(lens_z).value
Ds = cosmo.angular_diameter_distance(source_z).value
Dds= cosmo.angular_diameter_distance_z1z2(lens_z,source_z).value

# number of dimensions our problem has
parameters = ['xL0','yL0','ML0','eL0','PAL0','shear','shearangle','xoffS0','yoffS0','fluxS0','majaxS0','indexS0','axisratioS0','PAS0']
n_params = len(parameters)
# name of the output files
prefix = "chains/02-"

def prior(cube, ndim, nparams):
    cube[0] = cube[0]*2.0-1.5
    cube[1] = cube[1]*2.0-1.0
    cube[2] = cube[2]*(1e13-1e9)+1e9
    cube[3] = cube[3]*0.8
    cube[4] = cube[4]*180.0
    cube[5] = cube[5]*0.5
    cube[6] = cube[6]*180.0
    cube[7] = cube[7]*0.3+0.1
    cube[8] = cube[8]*0.3
    cube[9] = cube[9]*0.01
    cube[10] = cube[10]*0.06
    cube[11] = cube[11]*4.99+0.01
    cube[12] = cube[12]*0.9+0.1
    cube[13] = cube[13]*180.0
    
def log_likelihood(cube, ndim, nparams):
    xL0,yL0,ML0,eL0,PAL0,shear,shearangle,xoffS0,yoffS0,fluxS0,majaxS0,indexS0,axisratioS0,PAS0 = cube[0],cube[1],cube[2],cube[3],cube[4],cube[5],cube[6],cube[7],cube[8],cube[9],cube[10],cube[11],cube[12],cube[13]
    
    lens = [
    vl.SIELens(z=0.34,
    x={'value':xL0,'fixed':False,'prior':[-1.5,0.5]},
    y={'value':yL0,'fixed':False,'prior':[-1.,1.]},
    M={'value':ML0,'fixed':False,'prior':[1e9,1e13]},
    e={'value':eL0,'fixed':False,'prior':[0.,0.8]},
    PA={'value':PAL0,'fixed':False,'prior':[0,180]}),

    vl.ExternalShear(
        shear={'value':shear,'fixed':False,'prior':[0.,0.5]},
        shearangle={'value':shearangle,'fixed':False,'prior':[0.,180.]})]
    
    source = [
    vl.SersicSource(z=3.62, lensed=True,
    xoff={'value':xoffS0,'prior':[0.1,0.4]},
    yoff={'value':yoffS0,'prior':[0.0,0.3]},
    flux={'value':fluxS0,'prior':[0.0,0.01]}, # 5 mJy
    majax={'value':majaxS0,'prior':[0.,0.06]},
    index={'value':indexS0,'prior':[0.01,5.]},
    axisratio={'value':axisratioS0,'prior':[0.1,1.]},
    PA={'value':PAS0,'prior':[0.,180]})]
    
    
    lnL,dphases = 0., [[]]*len(data)
          
    mus = np.zeros(len(source))
           
    # Make a model of this field.
    immap,mags = vl.create_modelimage(lens,source,xmap,ymap,xemit,yemit,indices,Dd,Ds,Dds,sourcedatamap[0])
           
    # Filter non-zero magnifications (only relevant if sourcedatamap)
    mus[mags != 0 ] = mags[mags != 0]
    
    scaleamp[0] = True    # False->True in pass_priors in calc_likelihood.py
            
    # ... and interpolate/sample it at our uv coordinates
    interpdata = vl.fft_interpolate(data[0],immap,xmap,ymap,ug,scaleamp[0])
    
    # If desired, do model-cal on this dataset
    if modelcal[0]:
        modeldata,dphase = vl.model_cal(data[0],interpdata,modelcal[0][0],modelcal[0][1])
        dphases[i] = dphase
        lnL -= (((modeldata.real - interpdata.real)**2. + (modeldata.imag - interpdata.imag)**2.)/modeldata.sigma**2.).sum()
                  
    # Calculate the contribution to chi2 from this dataset
    else: lnL -= (((data[0].real - interpdata.real)**2. + (data[0].imag - interpdata.imag)**2.)/data[0].sigma**2.).sum()
    
    
    # Last-ditch attempt to keep from hanging
    if np.isnan(lnL): return -np.inf,[np.nan]
    
    return lnL

# run MultiNest
mcmcresult = pymultinest.run(LogLikelihood=log_likelihood, Prior=prior, n_dims=n_params, outputfiles_basename=prefix, verbose=True, importance_nested_sampling = True)


# store the parameter names:
import json
with open('%sparams.json' % prefix, 'w') as f:
    json.dump(parameters, f, indent=2)

# create analyzer object
a = pymultinest.analyse.Analyzer(n_params, outputfiles_basename = prefix)

s = a.get_stats()

json.dump(s, open(prefix + 'stats.json', 'w'), indent=4)

print('  marginal likelihood:')
print('    ln Z = %.1f +- %.1f' % (s['global evidence'], s['global evidence error']))
print('  parameters:')
for p, m in zip(parameters, s['marginals']):
	lo, hi = m['1sigma']
	med = m['median']
	sigma = (hi - lo) / 2
	if sigma == 0:
		i = 3
	else:
		i = max(0, int(-np.floor(np.log10(sigma))) + 1)
	fmt = '%%.%df' % i
	fmts = '\t'.join(['    %-15s' + fmt + " +- " + fmt])
	print(fmts % (p, med, sigma))

print('creating marginal plot ...')
data = a.get_data()[:,2:]
weights = a.get_data()[:,0]

#mask = weights.cumsum() > 1e-5
mask = weights > 3.5*1e-4

#corner.corner(data[mask,:], weights=weights[mask], 
#	labels=parameters, show_titles=True)
corner.corner(data[mask,:], labels=parameters)
pl.savefig(prefix + 'corner.pdf')
