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

seq = batman.Sequence(chain,False)
seq.set_init_quantity(startingQuantity)
seq.add_initial(0.,7200.,flux)
seq.add_after_last()
seq.check_steps()

tpoints = np.linspace(0.,86400.,num=240)
nuclideList = ['Fe54','Fe55','Fe56','Fe57','Fe58','Fe59','Co59']

activity, totalActivity = seq.solve_activity(tpoints)
print("%.0f sec: %.0f Bq"%(tpoints[-1],totalActivity[-1]))

seq.plot_quantity(tpoints,nuclideList,ymax=10**24,xlogScale=False,cmap='jet')
seq.plot_activity(tpoints,nuclideList,ymax=10**9,xlogScale=False,cmap='jet')

batman.Sequence.show()
