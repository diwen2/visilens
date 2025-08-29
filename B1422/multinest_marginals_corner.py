#!/usr/bin/env python
from __future__ import absolute_import, unicode_literals, print_function
__doc__ = """
Script that does default visualizations (marginal plots, 1-d and 2-d).

Author: Johannes Buchner (C) 2013-2019
"""
import numpy
from numpy import exp, log
import matplotlib.pyplot as plt
import sys, os
import json
import pymultinest
import corner

if len(sys.argv) != 2:
	sys.stderr.write("""SYNOPSIS: %s <output-root> 

	output-root: 	Where the output of a MultiNest run has been written to. 
	            	Example: chains/1-
%s""" % (sys.argv[0], __doc__))
	sys.exit(1)

prefix = sys.argv[1]
print('model "%s"' % prefix)
if not os.path.exists(prefix + 'params.json'):
	sys.stderr.write("""Expected the file %sparams.json with the parameter names.
For example, for a three-dimensional problem:

["Redshift $z$", "my parameter 2", "A"]
%s""" % (sys.argv[1], __doc__))
	sys.exit(2)
parameters = json.load(open(prefix + 'params.json'))
n_params = len(parameters)

a = pymultinest.Analyzer(n_params = n_params, outputfiles_basename = prefix)
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
		i = max(0, int(-numpy.floor(numpy.log10(sigma))) + 1)
	fmt = '%%.%df' % i
	fmts = '\t'.join(['    %-15s' + fmt + " +- " + fmt])
	print(fmts % (p, med, sigma))

print('creating marginal plot ...')
data = a.get_data()[:,2:]
weights = a.get_data()[:,0]
#mask = weights.cumsum() > 1e-5
mask = weights > 0
print(data[mask, :].shape)
modemax = -1
for i in range(len(s['modes'])):
    if a.get_best_fit()['parameters'] == s['modes'][i]['maximum']:
        modemax = i

bf = s['modes'][modemax]['maximum']
mean = s['modes'][modemax]['mean']
low1 = numpy.zeros(n_params)
high1 = numpy.zeros(n_params)
for i in range(n_params):
    #low1[i] = s['marginals'][i]['1sigma'][0]
    #high1[i] = s['marginals'][i]['1sigma'][1]
    low1[i] = s['modes'][modemax]['mean'][i] - s['modes'][modemax]['sigma'][i]
    high1[i] = s['modes'][modemax]['mean'][i] + s['modes'][modemax]['sigma'][i]

#figure = corner.corner(data, labels=parameters, quantiles=[0.16, 0.84], show_titles=True)
#figure = corner.corner(data,
figure = corner.corner(data[mask, :], weights=weights[mask], quantiles=[0.16, 0.5, 0.84],
        labels=parameters, show_titles=True, use_math_text=True)
axes = numpy.array(figure.axes).reshape((n_params, n_params))
#for yi in range(n_params):
#    for xi in range(yi):
#        ax = axes[yi, xi]
#        ax.axvline(low1[xi], color="k")
#        ax.axvline(high1[xi], color="k")

#for i in range(n_params):
#    ax = axes[i, i]
#    ax.axvline(low1[i], color="b", linestyle='dashed')
#    ax.axvline(high1[i], color="b", linestyle='dashed')

#corner.overplot_lines(figure, bf, color="orange")
#corner.overplot_points(figure, numpy.array([bf]), marker="s", color="orange")
#corner.overplot_lines(figure, mean, color="green")
#corner.overplot_points(figure, numpy.array([mean]), marker="s", color="green")
#corner.overplot_lines(figure, low1, color="k", linestyle='dashed')
#corner.overplot_points(figure, numpy.array([low1]), marker="s", color="k")
#corner.overplot_lines(figure, high1, color="k", linestyle='dashed')
#corner.overplot_points(figure, numpy.array([high1]), marker="s", color="k")
plt.savefig(prefix + 'corner.pdf')
plt.savefig(prefix + 'corner.png')
plt.close()


