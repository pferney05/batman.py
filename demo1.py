#!/Users/fernpa/anaconda3/bin/python

import numpy as np
import batman

startingQuantity = {
    'Fe54':0.0585 * 6.02e23,
    'Fe56':0.9180 * 6.02e23,
    'Fe57':0.0212 * 6.02e23,
    'Fe58':0.0028 * 6.02e23,
    }

flux = batman.Flux(fluxes=7.e11)

chain = batman.Chain()
chain.load_cross_sections('Xsection.csv')

time = np.array([0.,7200,7200.1,86400.])
power = np.array([7.e11,10.e11,0.,0.])
prob = batman.Continuous(chain, time, power, startingQuantity, fission=False)

tpoints = np.linspace(0.,86400.,num=240)

nuclideList = ['Fe54','Fe55','Fe56','Fe57','Fe58','Fe59','Co59']

activity, totalActivity = prob.solve_activity(tpoints)
print("%.0f sec: %.0f Bq"%(tpoints[-1],totalActivity[-1]))

prob.plot_quantity(tpoints,nuclideList,ymax=10**24,xlogScale=False,cmap='jet')
prob.plot_activity(tpoints,nuclideList,ymax=10**9,xlogScale=False,cmap='jet')

batman.Sequence.show()
