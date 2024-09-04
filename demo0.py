#!/Users/fernpa/anaconda3/bin/python

import numpy as np
import batman


startingQuantity = {
    'U238':0.993 * 6.02e23,
    'U235':0.007 * 6.02e23
    }

flux = batman.Flux(fluxes=0.)
chain = batman.Chain()

seq = batman.Sequence(chain,False)
seq.set_init_quantity(startingQuantity)
seq.add_initial()

tpoints = np.logspace(0,17,num=100)
nuclideList = ['U238','U234','Th230','Pb206','U235','Pa231','Pb207']

seq.plot_quantity(tpoints,nuclideList,ymax=10**24,xlogScale=True,cmap='jet',xscale=3600.*24*365.25)
seq.plot_activity(tpoints,nuclideList,ymin=10**-12,ymax=10**9,xlogScale=True,cmap='jet',xscale=3600.*24*365.25)

batman.Sequence.show()
